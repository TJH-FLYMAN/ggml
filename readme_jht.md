# ggml

## 总览与范围

GGML 是一个以 tensor 和计算图为核心的机器学习张量库。本文以当前仓库源码为准，介绍离线推理涉及的核心模块；具体 backend 实现只展开 CPU，不介绍训练、梯度和反向传播。

| 模块 | 核心对象/接口 | 职责 |
| --- | --- | --- |
| tensor 与算子 | `ggml_tensor`、`ggml_type`、`ggml_op` | 描述数据类型、shape、stride、view、算子及依赖关系 |
| context | `ggml_context`、`ggml_object` | 管理 tensor、graph 等对象的元数据内存，以及可选的内联 tensor data |
| computation graph | `ggml_cgraph` | 从输出 tensor 回溯依赖并建立前向计算图 |
| backend abstraction | backend、device、buffer type、buffer | 抽象设备、内存和计算接口；具体实现只展开 CPU backend |
| allocation | `ggml_tallocr`、`ggml_gallocr` | 分配 tensor，或根据图中生命周期规划和复用执行期内存 |
| scheduler | `ggml_backend_sched` | 选择节点 backend、切分子图、创建跨 backend copy、调用 gallocr 分配内存并驱动各 split 执行 |
| CPU execution | CPU backend、compute plan、threadpool | 将 graph 中的 op 分派给 CPU kernel 并行执行 |
| quantization | type traits、`ggml_quantize_*` | 定义量化块格式、数据转换和 CPU 量化计算能力 |
| GGUF | `gguf_context` | 读取、检查、修改和写出模型元数据及 tensor 数据 |

典型离线推理路径：

GGUF 加载模型
→ tensor/算子定义依赖
→ 构建前向 `ggml_cgraph`
→ allocator 绑定 backend buffer
→ scheduler 选择 backend、切图并处理必要的数据复制
→ 对应 backend 执行 split
→ 读取输出 tensor

本文完整介绍 scheduler 的通用多 backend 调度机制，但具体 backend 的内部实现只介绍 CPU。

## Tensor、数据布局与算子

`ggml_tensor` 不只是一个数据指针。它同时承担三种角色：

1. 描述逻辑数据：`type`、`ne[]` 和 `nb[]` 定义数据类型、形状和字节步长。
2. 定位物理数据：`buffer`、`data`、`view_src` 和 `view_offs` 描述数据存放位置及别名关系。
3. 表示计算节点：`op`、`op_params` 和 `src[]` 记录生成该 tensor 的算子及其输入依赖。

### `ggml_tensor` 核心字段

| 字段 | 含义 |
| --- | --- |
| `type` | tensor 的存储类型，如 F32、F16、BF16 或块量化类型 |
| `ne[GGML_MAX_DIMS]` | 各维元素数；当前 `GGML_MAX_DIMS` 为 4 |
| `nb[GGML_MAX_DIMS]` | 各维字节步长；第 0 维是最内层维度 |
| `buffer` | 管理 tensor 数据的 backend buffer；context 内联分配时可以为 `NULL` |
| `data` | tensor 数据地址；尚未分配时可以为 `NULL` |
| `op` | 生成该 tensor 的 `ggml_op`；普通输入或常量通常为 `GGML_OP_NONE` |
| `op_params` | 算子的内嵌参数区，例如 offset、stride、axis 或精度配置 |
| `src[]` | 算子依赖的源 tensor，最多 `GGML_MAX_SRC` 个 |
| `flags` | tensor 标志；离线推理主要关注 `GGML_TENSOR_FLAG_INPUT/OUTPUT` |
| `view_src`、`view_offs` | view 的底层 tensor 以及相对底层数据起点的字节偏移 |
| `name` | 调试、查找和 graph dump 使用的名称 |
| `extra` | backend 使用的扩展信息，核心代码不解释其具体布局 |

tensor metadata 与 tensor data 的生命周期并不总是绑定：释放 `ggml_context` 会释放其中的 tensor metadata，但 backend buffer 由相应的 buffer 生命周期管理。

### 数据类型、block 与大小

`ggml_type` 大致分为：

- 普通浮点和整数类型，例如 F32、F16、BF16、F64、I8、I16、I32、I64。
- 块量化类型，例如 Q、K-quant、IQ 和 TQ 系列。多个逻辑元素被编码在一个存储 block 中。

通用 type traits 记录 `type_name`、`blck_size`、`type_size`、是否量化以及参考转换函数。常用查询接口为：

- `ggml_blck_size(type)`：一个存储 block 包含的逻辑元素数；普通类型为 1。
- `ggml_type_size(type)`：一个完整 block 占用的字节数。仅当 block size 为 1 时，它才等于单个逻辑元素大小。
- `ggml_row_size(type, ne0)`：第 0 维一整行的字节数，要求 `ne0` 是 block size 的整数倍。
- `ggml_nbytes(tensor)`：结合 shape 和 stride 计算当前 tensor 布局覆盖的字节范围；非连续 tensor 不应简单使用“元素数 × 元素大小”。
- `ggml_get_type_traits(type)`：取得该类型的通用 traits。

具体量化 block 的字段布局、转换函数和 CPU dot kernel 在量化章节说明。

### Shape 与 stride

GGML tensor 固定保存四个维度，未使用的高维初始化为 1。`ne[0]` 是最内层维度，二维 tensor 通常可理解为 `ne[1]` 行、每行 `ne[0]` 个逻辑元素。

新建连续 tensor 时，默认 stride 为：

```text
nb[0] = ggml_type_size(type)
nb[1] = nb[0] * ne[0] / ggml_blck_size(type)
nb[i] = nb[i - 1] * ne[i - 1], i >= 2
```

对于 block size 为 1 的普通类型，元素 `(i0, i1, i2, i3)` 的字节偏移为：

```text
i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3]
```

量化 tensor 的第 0 维按 block 编码，不能用单个逻辑元素直接套用上述 `i0*nb[0]`；应按 block 和 row size 访问。

`ggml_is_contiguous()` 检查 stride 是否符合连续布局。transpose 或 permute 只交换 shape/stride 时通常会得到非连续 tensor；需要连续数据的算子可通过 `ggml_cont()` 创建实际的数据重排节点。

### Tensor 数据的来源

tensor data 常见有三种来源：

1. context 内联数据：`no_alloc == false` 时，普通非 view tensor 的 metadata 后可以紧跟数据区，此时 `buffer == NULL`。
2. backend buffer：`no_alloc == true` 时先只创建 metadata，之后由 backend allocator 设置 `buffer` 和 `data`。CPU buffer 中的 `data` 是主机可访问地址。
3. view：不申请独立数据区，通过 `view_src` 和 `view_offs` 复用底层 tensor 的存储。

创建嵌套 view 时，`ggml_new_tensor_impl()` 会把引用折叠到最底层 `view_src`，并累加 offset。因此 `view_offs` 是相对底层 tensor 的绝对字节偏移，而不是只相对上一层 view。

### View、reshape、copy 与 in-place

- `ggml_view_1d/2d/3d/4d`：指定新的 shape、stride 和 byte offset，共享底层数据。
- `ggml_reshape_*`：要求源 tensor 连续且元素总数不变，只改变 shape 并共享底层数据。
- `ggml_permute()` / `ggml_transpose()`：交换 `ne[]` 和 `nb[]`，不搬运底层数据。
- `ggml_cont()`：创建 `GGML_OP_CONT` 节点，执行后得到按目标 shape 连续存放的数据。
- `ggml_cpy(ctx, a, b)`：返回目标 `b` 的 view，并创建把 `a` 写入 `b` 的 `GGML_OP_CPY` 节点；不是让 `b` 改为引用 `a` 的存储。
- `_inplace` 算子：结果 tensor 是第一个输入的 view，但仍是带有 `op/src[]` 的独立 graph node。执行时会覆盖输入存储，必须确保该输入之后不再需要原值。

调用者必须保证 view 按最终 shape/stride 覆盖的 byte range 落在底层 tensor 数据范围内。view 的 shape 不需要逐维小于源 tensor；真正的约束是最终访问不能越界。`ggml_new_tensor_impl()` 会检查初始连续布局的 `data_size + view_offs`，而 `ggml_view_2d/3d/4d()` 随后可覆盖高维 stride，因此自定义 stride 的边界仍需调用者保证。

### 算子调用只构造计算关系

以 `ggml_add()` 为例，函数会检查输入 shape 是否可 repeat，创建结果 tensor，然后记录：

```text
result->op     = GGML_OP_ADD
result->src[0] = a
result->src[1] = b
```

此时只完成结果 shape、存储需求和依赖关系的描述；即使结果已经有 data 空间，其中也还没有有效计算结果。真正的数值计算发生在 graph 被 backend 执行时。

离线推理常见算子可按用途理解：

| 类别 | 代表 API |
| --- | --- |
| 元素级与激活 | `ggml_add`、`ggml_mul`、`ggml_scale`、`ggml_relu`、`ggml_silu` |
| reduction 与 normalization | `ggml_sum`、`ggml_mean`、`ggml_norm`、`ggml_rms_norm` |
| 矩阵运算 | `ggml_mul_mat`、`ggml_mul_mat_id`、`ggml_out_prod` |
| layout 与数据移动 | `ggml_dup`、`ggml_cpy`、`ggml_cont`、`ggml_reshape_*`、`ggml_view_*`、`ggml_permute` |
| 索引、mask 与 softmax | `ggml_get_rows`、`ggml_diag_mask_inf`、`ggml_soft_max` |
| 卷积与图像操作 | `ggml_im2col`、`ggml_conv_1d`、`ggml_conv_2d`、`ggml_pool_2d`、`ggml_upscale` |
| 序列与 attention | `ggml_rope`、`ggml_flash_attn_ext`、`ggml_ssm_scan`、`ggml_gated_linear_attn` |
| 自定义计算 | `ggml_map_custom1/2/3` |

不同 backend 不一定支持全部 `ggml_op`；scheduler 分配节点时会通过 device 的 `supports_op` 能力查询选择可执行的 backend。

## 1. GGUF
gguf模型解析后，元数据与 tensor info 保存在 `gguf_context`  

**1.1 GGUF文件结构**  
[ Header ] -> [ Metadata ] -> [ Tensor Info ] -> [ Tensor Data ]  

gguf-header   
- magic;           `GGUF` 的字节序标识，对应 `0x47 0x47 0x55 0x46`  
- version;         当前仓库支持到 GGUF v3  
- n_tensor;        tensor数量  
- n_kv;            metadata数量（kv 对数量）  

metadata   
- `gguf_metadata_kv_t metadata_kv[metadata_kv_count]`      
- key(string)  
- value  
    a. 非 array：按 `gguf_type` 读取单个值  
    b. array：先记录 `GGUF_TYPE_ARRAY`、数组元素类型、元素个数，再读取对应数组内容  

tensor_info   
- name_len(uint64) + name(string)   
- n_dims(uint32)  
- dims\[n_dims\](uint64)  
- datatype(uint32)    
- offset(uint64，指向 tensor data 区内偏移，按 alignment 对齐)    

Tensor_Data 
- offset : 读取完 tensor_info 后，文件位置需要先按 `alignment` 对齐，之后才是 tensor data 区起始 offset  
- size :按 `ggml_nbytes(tensor)` 计算，并对每个 tensor 做 alignment padding。量化类型有额外逻辑

**1.2 gguf_context**  
 `gguf_context` 保存：`version + kv + tensor_info + alignment + data_offset + data_size + data`  

**1.3 gguf模型加载**    
```c
// 1. 根据 model path 创建 gguf_context，同时可选地创建 ggml_context
std::string model_fname = "model.gguf";
struct ggml_context * tmp_ctx = nullptr;
struct gguf_init_params gguf_params = {
    /*.no_alloc   =*/ false,
    /*.ctx        =*/ &tmp_ctx,
};
// no_alloc = false 且 ctx != nullptr 时，会额外创建 ggml_context 并读入 tensor data
gguf_context * gguf_ctx_from_file = gguf_init_from_file(model_fname.c_str(), gguf_params);

// 2. 创建空 gguf_context
gguf_context * gguf_ctx_empty = gguf_init_empty();
```

`gguf_init_from_file_impl`  
1. 分配并初始化 `gguf_context`  
2. 读取 header、metadata、tensor info，并根据 `general.alignment` 计算 data 区 offset/size  
3. 如果 `gguf_params.ctx != nullptr`，则创建 `ggml_context`  
4. 若 `no_alloc == false`，读取整块 tensor data，并让各 tensor 的 `data` 指向相应 offset  
5. 若 `no_alloc == true`，只创建 tensor metadata，不读取 tensor data blob 到 `ggml_context`  

## 2. ggml_context  
`ggml_context` 通过 `ggml_object` 链表管理 context 中的对象；对象不只包括 tensor，也包括 graph、work buffer 等。  
model_data也视为ggml_object,并为ctx的头节点。+1 指 model_data对应的ggml_tensor  

**2.1 ggml_new_tensor**  
为每个 tensor 在 `ggml_context` 内存池分配对象。分配对象至少包括 `ggml_object + ggml_tensor metadata`；只有在非 view 且 `!no_alloc` 时，才会额外为 tensor data 分配空间。  

1. 为 `ggml_object` 分配内存  
在 `ggml_context` 的内存池中分配 object 对象内存
```c
struct ggml_object {
    size_t offs;   // object 数据区相对于 mem_buffer 的偏移
    size_t size;   // object 大小包含  struct ggml_tensor在内）  
    struct ggml_object * next;
    enum ggml_object_type type;
}
```
2. 为 `ggml_tensor` 分配内存  
`struct ggml_tensor * tensor_new = (char *) ctx->mem_buffer + obj->offs`  
3. 为 tensor_data 分配内存  
仅当 `view_src == NULL && !ctx->no_alloc` 时，tensor data 才会随同该 object 一起分配  

**2.2 ggml_context**  
获取 `ggml_context`  
```c
// no_alloc = true : 只分配 tensor metadata
// no_alloc = false: 普通 tensor 可在 context 内直接分配 data
bool no_alloc = false;
const size_t mem_size =
    no_alloc ?
    (n_tensors    )*ggml_tensor_overhead() :
    (n_tensors + 1)*ggml_tensor_overhead() + ctx->size;// +1对应tensor_data
struct ggml_init_params params = {
/*mem_size   =*/ mem_size,
/*mem_buffer =*/ nullptr,
/*no_alloc   =*/ no_alloc,
};
struct ggml_context * ctx = ggml_init(params);
```
 
`ggml_context` 中常见的几种内存使用方式：
1. 在 `gguf_init_from_file` 中创建 `ggml_context`（`ggml_ctx` 由加载逻辑创建）
   a. `gguf_init_params.no_alloc == false && gguf_init_params.ctx != nullptr`
      先读取整块 tensor data，再创建各 tensor metadata；各 tensor 的 `data` 根据 `info.offset` 指向这块连续数据 Abcbcbcbcbc  
   b. `gguf_init_params.no_alloc == true && gguf_init_params.ctx != nullptr`
      只创建 tensor metadata，tensor data 不读入该 `ggml_context`
2. 单独 `ggml_init()` 创建 `ggml_context`（适用于动态构建张量）
   a. `params.no_alloc = false`
      tensor metadata 与 tensor data 都可直接从 `ggml_context.mem_buffer` 中分配 ， tensor_num个[object_metadata tensor_metadata + tensor_data]内存中排列，每个张量元数据与数据连续存储  abcabcabc...abc 
   b. `params.no_alloc = true`
      此时张量元数据在 `ggml_ctx.mem_buffer` 中，但 `data` 通常为 `NULL`；后续可调用 `ggml_backend_alloc_ctx_tensors()` 为这些张量分配 `ggml_backend_buffer` 并绑定  


## 3.ggml_cgraph
graph 需要自定义构建，流程：  
- 分配 `ggml_cgraph` 内存
- 定义计算图，使用算子函数连接weight参数并创建中间计算节点
- 使用 `ggml_build_forward_expand()` 构建计算图
- 默认最大节点数由 `GGML_DEFAULT_GRAPH_SIZE=2048` 决定  

node 为计算节点，数组中的元素是指向 tensor 的指针  
leaf 一般是常量、输入、权重等不执行 op 的张量  
`ggml_cgraph_eval_order` 代表递归时 `src[]` 数组遍历正序或逆序，影响拓扑遍历顺序  


**3.1 内存分配**  
内存分配分为图张量内存分配器ggml_galloc_t和图内存申请ggml_new_graph  
- `ggml_new_graph` / `ggml_new_graph_custom`：为计算图对象本身分配内存  
- `ggml_gallocr`：为图中的张量分配实际后端内存  

`ggml_new_graph(ggml_ctx)`  
- 根据 size 计算 graph 所需内存大小，包含 `nodes / leafs / hash table` 等  
- 在 `ggml_context` 中新建 graph object  
- 计算 graph 内各数组在 object 数据区中的起始地址  
- 返回 `ggml_cgraph *`  


**3.2 定义计算节点**  
例如定义 `y = d * (a * w + b)`，初始化中间节点的 `src` 和 `op_type`  
```c
struct ggml_tensor * aw         = ggml_mul(ggml_ctx, a, w);
struct ggml_tensor * y          = ggml_add(ggml_ctx, aw, b);
struct ggml_tensor * res_tensor = ggml_mul(ggml_ctx, y, d);
```

**3.3 构建计算图**  
`ggml_visit_parents` 从结果张量为根节点开始，递归访问每个节点依赖的 `src[:]`，再把遍历到的张量分别放入 `ggml_cgraph.nodes` 和 `ggml_cgraph.leafs`  

`ggml_visit_parents(cgraph, res_tensor)` 判断逻辑  
- 从 `res_tensor` 开始，遍历所有依赖张量；`cgraph.order` 决定 `tensor.src[:]` 遍历正序或逆序,默认正序  
- `hash_set` 记录已访问张量，避免重复遍历  
- leafs: `tensor.op == GGML_OP_NONE` 且该 tensor 不是 param tensor
- leafs 以外都为 node  

以 gpt-2 为例，从结果张量回溯构建图，详见 `docs/ggml_visit_parents.jpg`  
`examples/gpt-2/main-backend.cpp:471-535`
DFS添加顺序为
leaf(h0/attn/c_attn/w) 
-> leaf(wte) -> leaf(embd) -> node(embeddings) -> leaf(wpe) -> leaf(position) -> node(pos_embeddings) -> node(inpL) 
-> node(norm_eps_res) -> leaf(h0/ln_1/g) -> node(norm_ln_1_g) -> leaf(h0/ln_1/b) -> node(norm_ln_1_b) 
-> node(attw_mul_res) -> leaf(h0/attn/c_attn/b) -> node(attw_mul_add_res) -> node(Kcur) -> node(k) -> node(k (copy of Kcur))


## 4.ggml_backend
ggml 后端大体可按下面的层次理解：

Backend Registry -> Device -> Backend  
Backend Registry -> Device -> Buffer Type -> Buffer  

Backend Registry 管理多个 Backend Registration  
每个 Registration 包含一个或多个 Device  
Device 可初始化 Backend 实例  
Device 提供 Buffer Type  
Buffer Type 负责分配具体的 Buffer  

以 cpu 后端为例  
cpu后端注册属于顶层接口，cpu_reg.iface包含获取cpu设备attr查询（name 、count）以及获取cpu_device实例  
cpu设备属于中间层。cpu_reg.device.iface包含device attr查询、后端初始化、获取Buffer Type  
cpu后端属于底层，cpu_reg.device.iface.init_backend.iface包含图创建、执行、tensor计算同步等 backend 实例负责图执行、tensor 读写、同步等  
cpu_reg -> cpu_device -> cpu_backend + cpu_backend_buffer_type_t  

各实例关系:
- cpu_reg 通过 iface.get_device 获取 cpu_device 实例；
- cpu_device 包含 reg 字段指向 cpu_reg（设备归属于后端注册） 
- cpu_device 通过 iface.init_backend 创建 cpu_backend 实例 
- cpu_backend 包含 device 字段指向 cpu_device（后端实例归属于设备）
- 双向引用, 下层结构通过指针反向引用上层，便于访问父级资源
具体关系查看docs/ggml_backend.jpg docs/ggml_backend.png

**4.0.1 ggml_backend_registry**   
后端通过注册机制加入全局注册表ggml_backend_registry，由全局注册表统一管理，允许运行时动态加载后端，也支持静态注册。  

获取注册表
get_reg()函数，返回静态全局注册表对象  
```
static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    return reg;
}
```

枚举后端reg ggml_backend_reg_count、ggml_backend_reg_get  
枚举dev     ggml_backend_dev_count、ggml_backend_dev_get 

**4.0.2 动态加载**  
通过 `ggml_backend_load` 加载单个后端动态库并注册,查找符号 ggml_backend_score（0 表示当前系统不支持该后端）和 ggml_backend_init，调用后者获取 ggml_backend_reg_t 并注册    

```c
#include "ggml-backend.h"
// 加载单个 so 后端
ggml_backend_load(so_path);
// 批量加载
ggml_backend_load_all_from_path(folder_path);
// 自动加载
ggml_backend_load_all();
```

ggml_backend_load_all自动对指定路径[./ , exec_path]中的libggml-{backend_name}-*.so评分，获取best backend，so保留dlopen句柄  

其中动态库编译时需使用宏 GGML_BACKEND_DL_IMPL(backend_reg_fn) 导出 ggml_backend_init 符号   

**4.0.3 静态注册**  
各后端实现 `ggml_backend_*_reg()` 函数，返回静态 backend reg 对象。  
注册表初始化时会收集后端、设备并加入容器。  

注意：  
- `backends` 容器里存的是 backend reg，而不是 backend 实例  
- `devices` 容器里存的是 device 对象  
- 实际执行用的 backend 仍然需要通过 `ggml_backend_dev_init()` 从 device 创建  

**4.1 ggml_backend_device**  
`ggml_backend_device` 连接后端注册表与实际执行 backend。对外通常通过接口函数访问，而不是直接依赖内部 struct 字段。  

获取 device 的常见方式：
```c
// 1. 枚举获取
size_t count = ggml_backend_dev_count();
for (size_t i = 0; i < count; i++) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
}

// 2. 根据 backend_name 获取
const char * backend_name = "CPU";
ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name);

// 3. 根据 backend_type 获取
ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);

// 4. 根据 backend_reg_t 获取
size_t dev_count = ggml_backend_reg_dev_count(reg);
for (size_t i = 0; i < dev_count; i++) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
}
```


**4.2 ggml_backend**  
`ggml_backend` 是 GGML 中用于抽象不同硬件执行后端的核心接口。它和 device 的关系是：device 负责创建 backend，backend 负责实际执行。  

**4.2.1 ggml_backend_cpu**  
获取 cpu backend  
```c
// 方式1: 直接调用专用接口
ggml_backend_t backend1 = ggml_backend_cpu_init();

// 方式2: 获取 dev，根据 dev 初始化 backend
const char * backend_name = "CPU";
ggml_backend_dev_t dev_by_name = ggml_backend_dev_by_name(backend_name);
ggml_backend_t backend2 = ggml_backend_dev_init(dev_by_name, nullptr);

// 方式3: 先动态加载，再按类型获取
ggml_backend_load_all();
ggml_backend_dev_t dev_by_type = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
ggml_backend_t backend3 = ggml_backend_dev_init(dev_by_type, nullptr);
```

**4.3 ggml_backend_buffer_type**
buffer_type = buffer 描述符（buffer descriptor） = 某类 backend 内存的分配策略/内存类型  
它描述内存类型、对齐要求，以及 backend 上如何分配 buffer  

每个 buffer type 通常提供这些能力：
- `get_name`
- `alloc_buffer`
- `get_alignment`
- `get_max_size`
- `get_alloc_size`
- `is_host`

获取 buft，以 cpu 为例：
```c
// backend -> 默认 buft
ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
```


**4.4 ggml_backend_buffer**
实际存放数据的 buffer。一个 buffer 可以承载多个 tensor。  

常见接口包括：
- `ggml_backend_buffer_get_base`
- `ggml_backend_buffer_get_size`
- `ggml_backend_buffer_get_type`
- `ggml_backend_buffer_set_usage`

根据 `ggml_ctx` 为其中 tensor 分配 backend buffer：
```c
ggml_backend_t backend = ggml_backend_cpu_init();

struct ggml_init_params pdata = {
    /*mem_size   =*/ mem_size,
    /*mem_buffer =*/ nullptr,
    /*no_alloc   =*/ true,
};

struct ggml_context * ggml_ctx = ggml_init(pdata);

// 为 context 中已有 tensor 统一分配 backend buffer
// 1. get buftype + buftype.iface.alloc_buffer for all tensor
// 2. combine all ggml_backend_buffer_t to one ggml_backend_buffer_t
// 3. ggml_backend_buffer_t以及addr回流至tensor，并init tensor buf（if required)
ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ggml_ctx, backend);
```

## 5.ggml_gallocr
gallocr（graph allocator）用于 graph 在不同后端上的张量内存管理。  
它通过预先规划内存布局来尽量复用图中的中间张量内存。  

`ggml_backend_alloc_ctx_tensors` 分配的是 context 中 tensor 的存储 buffer（权重）；`ggml_gallocr` 主要处理图执行阶段的张量分配（workspace buf）。  

ggml_gallocr 的典型使用流程：
- `ggml_gallocr_new`
- `ggml_gallocr_reserve`
- `ggml_gallocr_alloc_graph`
- `ggml_gallocr_get_buffer_size`
- 整个流程从 buffer type 出发，先创建图内存分配器 gallocr  
- 然后用最坏情况 graph 估算执行期中间张量需要的显存/内存，并 reserve  
- 真正执行前再把 offset/base_ptr 回写到 graph 里的 tensor 上  

**5.0 角色边界**  
可以把 ggml 里的 tensor 数据分成两类看：  
1. 静态 tensor / 权重  
   一般通过 `ggml_backend_alloc_ctx_tensors` 或模型加载阶段直接拥有自己的 buffer  
2. graph 执行期中间 tensor  
   这部分生命周期短，适合交给 `ggml_gallocr` 统一规划、复用和回收  

所以 `ggml_gallocr` 不是“给所有 tensor 分配内存”，而是“给这次 graph 执行过程中需要的 tensor 布局做一次整体规划”。  

**5.1 核心结构**  
1. `ggml_gallocr`  
   图级分配器，管理多个 `buft/buffer`，并记录 graph 中每个 node/leaf 最终落到哪个 buffer、哪个 offset。  
   关键成员：
   - `bufts[]`：每个后端对应的 buffer type
   - `buffers[]`：真正分配出来的 backend buffer
   - `buf_tallocs[]`：每个 buffer 对应的动态分配器
   - `hash_set + hash_values[]`：按 tensor 指针建立哈希表，记录该 tensor 的生命周期和分配信息
   - `node_allocs[] / leaf_allocs[]`：把 reserve 阶段得到的 `(buffer_id, offset, size_max)` 固化下来，供后续 alloc_graph 直接回写  

2. `ggml_dyn_tallocr`  
   单个 buffer 内部的动态分配器。  
   它不关心 graph，只关心“当前 buffer 里哪些 free block 可用”。  
   关键成员：
   - `alignment`
   - `free_blocks[]`
   - `n_free_blocks`
   - `max_size`：本轮规划过程中用到过的最大地址，用于决定最终 buffer 要申请多大  

3. `hash_node`  
   gallocr 在 reserve 期间给每个 tensor 维护的生命周期状态：
   - `n_children`：还有多少后继节点会继续使用它
   - `n_views`：还有多少 view 依赖它
   - `buffer_id`
   - `offset`
   - `allocated`：当前是否还占着 gallocr 自己管理的内存  

4. `node_alloc / leaf_alloc`  
   reserve 阶段的结果快照。  
   后面 `ggml_gallocr_alloc_graph` 不再重新推导布局，而是直接使用这些快照给 tensor 初始化 backend 地址。  

**5.2 ggml_gallocr_new_n**
根据一个或多个 buffer type 创建 gallocr
```c
// 获取一个 gallocr
ggml_gallocr_t allocr = ggml_gallocr_new(buft);
// 获取 n_bufs 个 buffer type 对应的 gallocr
ggml_gallocr_t allocr_n = ggml_gallocr_new_n(bufts, n_bufs);
```

`ggml_gallocr_new_n` 的关键点：  
- `galloc->bufts[i] = bufts[i]`
- 为每个 buft 找对应的 `ggml_dyn_tallocr`
- 如果多个 backend 使用的是同一个 buft，则共享同一个 `ggml_dyn_tallocr`  

这意味着：  
不同 backend 只要底层 buffer type 相同，就可以复用同一套 buffer 规划逻辑，不需要重复维护 allocator。  

**5.3 ggml_gallocr_reserve_n**  
根据 graph 规划计算节点所需内存，并为后续分配做预留。  
这一步常用于“先用最坏情况图估算一次内存”，避免运行中频繁重分配。
主流程如下：  
1. 根据 `graph->n_nodes + graph->n_leafs` 计算最小 hash size，再额外加 25% 冗余  
2. reset 每个 `ggml_dyn_tallocr`  
   reset 后初始只有一个超大 free block，offset=0  
3. 调用 `ggml_gallocr_alloc_graph_impl` 做一次“模拟分配”  
4. 根据模拟结果，把每个 node/leaf 的 `(buffer_id, offset, size_max)` 写入 `node_allocs[] / leaf_allocs[]`
5. 根据各 `ggml_dyn_tallocr.max_size` 的结果，真正申请或扩容 backend buffer  

`ggml_gallocr_alloc_graph_impl` 是 reserve 的核心。它做的不是给 tensor 立刻绑地址，而是先把整张图走一遍，算出每个 tensor 应该落在哪个 offset。  

它内部大致分三段：  
1. 先处理 leaf  
   叶子节点先尝试分配，避免后续节点覆盖掉它们  
2. 再遍历 node 统计生命周期  
   - 如果 node 是 view，则给 `view_src.n_views++`
   - 对每个 `src` 做 `n_children++`
   - 显式 input tensor 会提前 allocate  
3. 按拓扑顺序正式“模拟执行”  
   - 先确保父节点已分配
   - 再给当前 node 分配
   - 当前 node 用完父节点后，把父节点 `n_children--`
   - 若某父节点 `n_children == 0 && n_views == 0`，则把它占用的内存回收到 free block 池中  

`ggml_gallocr_allocate_node` 的分配策略：  
- 如果 op 支持 inplace，则优先尝试复用父节点 buffer  
- 只有当父节点满足“仅有一个 child、没有 view、layout 相同、且不是 output”时，才允许复用  
- 否则走 `ggml_dyn_tallocr_alloc` 新分配 offset  

`ggml_dyn_tallocr_alloc` 的策略可以理解成 best-fit：  
- 优先在已有 free block 中找一个 `>= size` 且最接近 size 的块  
- 如果前面的 free block 都不合适，再使用最后一个 free block 兜底  
- 分配成功后切掉这段 free block，更新 `max_size`  

`ggml_dyn_tallocr_free_tensor` 释放时会把空间插回 `free_blocks[]`，并尝试和前后相邻块合并。  
所以 gallocr 的“内存复用”本质上是：  
- 拓扑执行时按生命周期回收中间 tensor  
- 后续 tensor 再从 free block 池里复用这段空间  

**5.4 ggml_gallocr_alloc_graph**  
根据 reserve 阶段的规划，为 graph 中张量设置 `data` 指针或对应 backend buffer 内地址。
这一步不再重新规划，只负责把 reserve 阶段记录下来的布局真正绑定到 tensor 上。  

流程如下：  
1. 先检查当前 graph 是否需要重新 reserve  
   - 若 `n_nodes/n_leafs` 变化了，或者某 tensor 当前大小超过了 reserve 时记录的 `size_max`，则认为规划失效  
2. 单 buffer 场景下，可自动重新 `ggml_gallocr_reserve`  
3. 多 buffer 场景下，不能自动重排，需要外部先显式调用 `ggml_gallocr_reserve_n`  
4. reset backend buffer  
5. 遍历 leafs 和 nodes，调用 `ggml_gallocr_init_tensor` 回写地址  

`ggml_gallocr_init_tensor` 有两种主要情况：  
1. 普通 tensor  
   - `base = ggml_backend_buffer_get_base(buffer)`
   - `addr = base + offset`
   - `ggml_backend_tensor_alloc(buffer, tensor, addr)`  
2. view tensor  
   - 不自己占有新内存
   - 通过 `ggml_backend_view_init(tensor)` 建立对源 tensor 的 view 关系  

所以 `reserve` 决定布局，`alloc_graph` 只是把布局落到 tensor 上。  

**5.5 数据流向**
数据可大致分为权重和中间计算结果：  
1. 权重 / 静态 tensor  
   `ggml_init_params.no_alloc = false` 时，可直接放在 `ggml_context.mem_buffer` 中  
   `no_alloc = true` 时,mem_buffer只持有info信息，权重由 `ggml_backend_alloc_ctx_tensors` 为这些 tensor 分配 backend buffer  
2. 中间计算结果  
   `ggml_gallocr` 在图执行阶段按需规划和复用  

一次完整调用链可以理解为：  
```c
ggml_gallocr_t galloc = ggml_gallocr_new_n(bufts, n_bufs);
ggml_gallocr_reserve_n(galloc, graph, node_buffer_ids, leaf_buffer_ids); // 只规划，不改 graph
ggml_gallocr_alloc_graph(galloc, graph); // 把 offset/base_ptr 回写到 tensor
size_t sz = ggml_gallocr_get_buffer_size(galloc, buffer_id);
```

`ggml_gallocr_get_buffer_size` 返回的是对应 backend buffer 的真实 size。  
如果多个 backend 共享了同一个底层 buffer，它只在第一次出现时返回 size，避免重复统计。  

## 6.ggml_backend_sched
后端调度用于分配节点计算。它会把整个 graph 划分成多个 subgraph，每个子图分配给一个 backend，并负责不同 backend 之间的数据流转。  

调用流程如下：
- 定义计算图时，可用 `ggml_backend_sched_set_tensor_backend` 指定 tensor 的 backend
- 创建调度器 `ggml_backend_sched_new`
- 遍历graph,切分subGraph,记录切分子图时需要拷贝的tensor( hv_tensor_copies )
- 预留 graph 所需内存 `ggml_backend_sched_reserve`
- 计算时调用 `ggml_backend_sched_graph_compute(sched, graph)`  

**6.1 ggml_backend_sched_new**  
创建调度器时，需要传入 backend 数组。当前实现中：
- backends 按优先级顺序排列（从高到低且backends[-1] = cpu_backend）
- 最后一个 backend 必须是 CPU backend
- 调度器内部会创建 `ggml_gallocr`

`sched_new` 初始化时会分配几类关键数组：  
1. `hv_tensor_backend_ids`  
   hash 表大小，默认初始值全为 `-1`，表示“这个 tensor 还没被分配 backend”  

2. `hv_tensor_copies`  
   单流水线时大小 = `hash_set.size * n_backends`  
   并行流水线时大小 = `hash_set.size * n_backends * n_copies`  
```c
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
```
由tensor_id_copy可知hv_tensor_copies的内存布局，以两个tensor(tensor_a tensor_b)、两个backend为例 (hash_set.size = 2, n_backends = 2)  

hv_tensor_copies内存布局(2 张量a b 、2 后端 cpu gpu)  
if parallel  
- size=16（2×2×4），按 [a_cpu_cp0~3,a_gpu_cp0~3,b_cpu_cp0~3,b_gpu_cp0~3] 排列  

else  
- size=4（2×2×1），按 [a_cpu,a_gpu,b_cpu,b_gpu] 排列  

3. `context_buffer_size`  
   用于内部临时 `ggml_context`，存放 split 过程中构造出来的 copy tensor / dependency tensor / graph view metadata。  
   大小按最坏情况估算：  
- graph_size * subGraph_max_input_num * 2 * sizeof(struct ggml_tensor)  ，n个node,切分n个subgraph,一个node最多6输入6输出  
- ggml_graph_overhead_custom(graph_size, false);  graph大小  

4. `node_backend_ids / leaf_backend_ids`  
   记录“复制后的 graph”中每个 node/leaf 应该由哪个 backend/buft 来分配  

5. `prev_node_backend_ids / prev_leaf_backend_ids`  
   用来判断这次 split 后 backend/buft 是否变了，如果变了，说明上一轮的 gallocr 规划可能失效，需要重新 reserve  

6. `splits / splits_capacity`  
   初始默认 16 个 split，后续按需翻倍扩容  

7. `galloc = ggml_gallocr_new_n(sched->bufts, n_backends)`  
   scheduler 自己并不直接做 offset 分配，而是把切好的 graph 交给 gallocr 去做内存布局  

**6.2 核心结构**  
1. `ggml_backend_sched_split`  
   表示一个连续子图：
   - `backend_id`
   - `i_start / i_end`
   - `inputs[]`：这个 split 运行前需要从别的 backend copy 进来的 tensor
   - `graph`：原 graph 的一个 view  

2. `ggml_backend_sched`  
   它同时维护三套信息：  
   - tensor 到 backend 的映射：`hv_tensor_backend_ids`
   - tensor 的跨 backend copy：`hv_tensor_copies`
   - split 结果及其 graph view：`splits[] + graph`  

所以 sched 的职责不是“真正执行一个 op”，而是：  
- 决定每个 node 放哪一个 backend  
- 在 backend 边界处创建 copy tensor  
- 组织 split 图  
- 调 gallocr 给 split 图分配内存  
- 最后按 split 顺序驱动各 backend 执行  

**6.3 ggml_backend_sched_split_graph**  
切分子图时，核心工作包括：
- 记录 leaf/node 对应的 backend id
- 根据 op 支持情况、tensor 所在 backend、优先级等信息决定节点归属
- split与split.input的backend不同时，插入cp_tensor(仅backend不同)。相邻子图 backend 不同

`ggml_backend_sched_split_graph` 大体可以分成 5 个 pass：  

1. pass1: 根据 tensor 当前所在位置做初始 backend 归属  
   - 预分配 tensor / weight tensor 优先跟随自己已有的 buffer/backend
   - graph input 默认放到最后一个 backend，也就是 CPU backend  

2. pass2: 扩散 backend 归属  
   - 先向下、再向上扩散高优先级 backend（通常是 GPU）
   - 再扩散剩余 backend  
   目的不是立刻得到最优解，而是尽量把相邻、兼容的 node 拉到同一个 backend，减少 split 和 copy  

3. pass3: 升级到更高优先级 backend  
   如果一个 node 当前所在 backend 的 buft 和更高优先级 backend 相同，且后者也支持该 op，则尝试把 node 升上去  
   典型例子就是多个 backend 共享 host buffer type 时，优先选优先级更高的 backend  

4. pass4: 给剩余 src/view_src 补 backend  
   view 一定跟随 `view_src`  
   其它尚未分配的 src，一般继承当前 node 的 backend  

5. pass5: 真正切 split，并创建 copy tensor  
   - 当 backend 发生切换，或者当前 split 输入过多、权重 backend 不兼容时，开始一个新的 split  
   - 如果 `src_backend_id != cur_backend_id` 且目标 backend 不能直接使用这个 src 的 buffer，则创建 `tensor_copy`
   - `tensor_copy` 用 `ggml_dup_tensor_layout` 创建，只复制 layout，不复制数据本身
   - 后续把 node 的 `src[j]` 改成这个 copy tensor  

切 split 完成后，scheduler 还会重建一份 `sched->graph`：  
- 把 split 的输入 copy tensor 插到 graph 前面，确保 gallocr 分配时它们先被看到  
- 再把原 graph 中属于该 split 的 node 依次放进去  
- 最后把原 graph 的 leaf 补进去  

因此 `sched->graph` 不是用户原始 graph，而是“已经插入 copy tensor、已经按 split 重组过的 graph”。  

**6.4 ggml_backend_sched_reserve**  
`sched_reserve` 做的是：  
1. 先 `ggml_backend_sched_split_graph(sched, measure_graph)`  
2. 同步所有 backend，避免上一轮异步执行仍在占用内存  
3. 调 `ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)`  
4. reserve 完后 reset scheduler，等待下一张真实 graph  

所以 scheduler 的 reserve，本质上还是“先 split，再把 split 结果交给 gallocr 做多 buffer 规划”。  

**6.5 ggml_backend_sched_alloc_graph**  
执行前的分配过程：  
1. `ggml_backend_sched_split_graph(sched, graph)`  
2. `ggml_backend_sched_alloc_splits(sched)`  

`ggml_backend_sched_alloc_splits` 内部会比较：
- `node_backend_ids` vs `prev_node_backend_ids`
- `leaf_backend_ids` vs `prev_leaf_backend_ids`  

如果 backend/buft 发生变化，或者 `ggml_gallocr_alloc_graph` 失败，则：  
- 先同步 backend
- 再重新 `ggml_gallocr_reserve_n`
- 然后重新 `ggml_gallocr_alloc_graph`  

这一步完成后，`sched->graph` 中所有 split tensor、copy tensor、原始 node tensor 都已经拿到了各自 backend buffer 内的地址。  

**6.6 ggml_backend_sched_graph_compute**  
最终计算通过 `ggml_backend_sched_compute_splits` 驱动：  
1. 依次取出每个 split  
2. 先处理 split 输入  
   - 若输入来自别的 backend，先 copy 到当前 split backend 对应的 `tensor_copy`
   - 用户输入 tensor 会优先同步后再立即拷贝，避免用户提前覆盖数据  
3. 调 `ggml_backend_graph_compute_async(split_backend, &split->graph)` 执行当前子图  
4. 若开启 pipeline parallel，则通过 `events[backend][copy]` 协调不同 copy 槽位的覆盖与复用  
5. 全部 split 执行完成后，`cur_copy = (cur_copy + 1) % n_copies`  

所以 scheduler 的运行时数据流可以概括为：  
`原始 tensor -> 按 backend 归属切 split -> 必要时生成 tensor_copy -> gallocr 给 split 图分配地址 -> 每个 split 在自己的 backend 上执行`  

**6.7 gallocr 与 sched 的关系**  
1. `sched` 负责“逻辑切图”  
   - 哪个 node 属于哪个 backend
   - 哪些输入需要 copy
   - split 图长什么样  
2. `gallocr` 负责“物理布局”  
   - 每个 split 图里的 tensor 放到哪个 buffer
   - 在 buffer 里的 offset 是多少
   - 哪些中间 tensor 可以复用同一段空间  
  
`sched` 解决“谁算、算哪一段、输入从哪来”；`gallocr` 解决“这段图里的 tensor 具体放到哪块内存”。  


可参考：
https://blog.rickyyel.org/blog/ggml-source-code-brief

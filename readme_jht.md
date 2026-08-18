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

## `ggml_context` 与对象内存池

`ggml_context` 是一个固定容量的 arena。它管理内存池、分配策略和 `ggml_object` 链表，tensor、graph 和 work buffer 都以 object 的形式顺序放入该内存池。context 不会自动扩容，也不支持单独释放其中某个 tensor 或 graph。

### Context 状态与内存所有权

`ggml_context` 的核心状态包括：

- `mem_size`、`mem_buffer`：arena 的容量和起始地址。
- `mem_buffer_owned`：内存池是否由 GGML 内部分配。
- `no_alloc`：新建普通 tensor 时是否跳过 tensor data 分配。
- `n_objects`、`objects_begin`、`objects_end`：object 数量和单向链表首尾。

`ggml_init()` 根据 `ggml_init_params` 初始化 context：

- `mem_buffer == NULL`：GGML 分配按 `GGML_MEM_ALIGN` 对齐的内存，`mem_buffer_owned = true`。
- `mem_buffer != NULL`：使用调用者提供的内存，调用者必须保证地址满足对齐要求；context 不负责释放这块内存。
- `no_alloc == true`：普通 tensor object 只包含 metadata，`data` 初始通常为 `NULL`。
- `no_alloc == false`：普通、非 view tensor 可在 metadata 后内联分配 data。

### `ggml_object` 与 arena 布局

`ggml_object` 是内存池中每段对象的头部：

```c
struct ggml_object {
    size_t offs;                  // payload 相对 ctx->mem_buffer 的偏移
    size_t size;                  // 按 GGML_MEM_ALIGN 对齐后的 payload 大小
    struct ggml_object * next;
    enum ggml_object_type type;
};
```

`size` 不包含 `ggml_object` 头本身。object 在 `mem_buffer` 中依次追加：

```text
[object header][aligned payload][object header][aligned payload] ...
```

不同 object type 的 payload 为：

```text
tensor:      [ggml_tensor][optional inline data][padding]
graph:       [ggml_cgraph + nodes/leafs/hash arrays][padding]
work buffer: [raw bytes][padding]
```

`obj->offs` 始终指向 payload，因此 tensor metadata 地址为：

```c
struct ggml_tensor * tensor =
    (struct ggml_tensor *) ((char *) ctx->mem_buffer + obj->offs);
```

### `ggml_new_tensor_impl()` 的分配过程

新建 tensor 的主要步骤是：

1. 根据 type 和 shape 计算连续布局的 `data_size`。
2. 若是嵌套 view，将 `view_src/view_offs` 折叠到最底层 tensor。
3. 仅当 `view_src == NULL && !ctx->no_alloc` 时，把 `data_size` 加入 object payload。
4. 调用 `ggml_new_object()` 分配并对齐 payload。
5. 在 `obj->offs` 初始化 `ggml_tensor`，设置 shape、stride、view 和 data。
6. 将 object 追加到 context 链表并增加 `n_objects`。

context 容量不足时不会增长内存池，因此 `mem_size` 必须在构图前估算充分。

### `mem_size` 估算

`ggml_tensor_overhead()` 返回一个 tensor object 的固定开销，即 object header 与 `ggml_tensor` metadata 之和，不包含 tensor data。

如果 context 只保存 `n_tensors` 个 tensor metadata，可按下面的方式初始化：

```c
const size_t mem_size = n_tensors * ggml_tensor_overhead();

struct ggml_init_params params = {
    .mem_size   = mem_size,
    .mem_buffer = NULL,
    .no_alloc   = true,
};

struct ggml_context * ctx = ggml_init(params);
```

使用 context 内联 data 时，每个普通非 view tensor 还需要增加：

```text
GGML_PAD(data_size, GGML_MEM_ALIGN)
```

如果同一个 context 中还要创建 graph，需要再计入 `ggml_graph_overhead()`，或用 `ggml_graph_overhead_custom()` 按自定义 graph size 计算。view 只需要 tensor overhead，不分配自己的 data。

### 常见使用模式

1. metadata-only context
   - `no_alloc = true`。
   - 先构造 tensor metadata，之后由 `ggml_backend_alloc_ctx_tensors()` 或 graph allocator 绑定 backend buffer。

2. context 内联数据
   - `no_alloc = false`。
   - 普通非 view tensor 的 metadata 与 data 位于同一个 object payload 中。
   - 常用于小型 CPU tensor、测试或不需要 backend allocator 的工具代码。

3. GGUF 创建的 context
   - `params.ctx != NULL && no_alloc == false` 时，加载器先创建一个 `GGML_TYPE_I8` tensor 保存整个 GGUF tensor-data blob，再临时切换为 `no_alloc` 创建模型 tensor metadata，并让各 tensor 的 `data` 指向 blob 内的对应 offset。
   - `params.ctx != NULL && no_alloc == true` 时，只创建模型 tensor metadata，不读取 tensor-data blob。

因此 GGUF 加载代码中 `(n_tensors + 1) * ggml_tensor_overhead()` 的 `+1` 是 data-blob tensor，不是通用 context 规则。

### Reset 与释放

- `ggml_reset(ctx)`：把 object 计数和链表首尾重置为空，以便从 arena 起点复用内存；它不会清零内存，旧 tensor/graph 指针不能继续使用。
- `ggml_free(ctx)`：释放 context 本身；只有 `mem_buffer_owned == true` 时才释放 `mem_buffer`。
- context 不拥有通过独立 backend allocator 创建的 backend buffer。backend buffer 需要通过对应的 buffer API 单独释放。

## 前向计算图 `ggml_cgraph`

算子函数先把依赖关系记录在 tensor 的 `op/src[]` 中，`ggml_cgraph` 再从一个或多个输出 tensor 回溯这些依赖，形成 backend 可以按拓扑顺序执行的前向图。

### Graph 保存的状态

推理 graph 的核心状态包括：

- `size`：`nodes[]` 和 `leafs[]` 各自可容纳的最大 tensor 数。
- `n_nodes`、`nodes[]`：需要 backend 执行的计算 tensor。
- `n_leafs`、`leafs[]`：没有生成算子的输入、权重或常量 tensor。
- `visited_hash_set`：记录已经访问的 tensor，避免共享依赖重复加入。
- `order`：内部 DFS 遍历 `src[]` 的顺序，默认从低索引到高索引。

graph 只保存 tensor 指针，不拥有 tensor metadata 或 tensor data。graph、tensor metadata 和它们所在的 context 必须在构图和执行期间保持有效。

### Graph object 的内存

离线推理通常创建：

```c
struct ggml_cgraph * graph =
    ggml_new_graph_custom(ctx, graph_size, false);
```

`ggml_new_graph(ctx)` 使用 `GGML_DEFAULT_GRAPH_SIZE`，当前值为 2048。推理 graph 的 object payload 包含：

```text
[ggml_cgraph]
[nodes[size]]
[leafs[size]]
[visited hash keys]
[visited bitset]
```

visited hash 的容量根据 `size * 2` 计算，因为它需要同时容纳 nodes 和 leafs。`ggml_graph_overhead()` 返回默认 graph 的 context 开销，`ggml_graph_overhead_custom(graph_size, false)` 返回自定义推理 graph 的开销。

`ggml_new_graph*()` 只分配 graph object 本身；图中 tensor data 的 backend 内存由后面的 allocator 负责。

### 从算子关系构建 graph

例如先定义：

```c
struct ggml_tensor * aw  = ggml_mul_mat(ctx, w, a);
struct ggml_tensor * sum = ggml_add(ctx, aw, b);
struct ggml_tensor * out = ggml_mul(ctx, sum, d);
```

这时 `aw/sum/out` 已记录 `op/src[]`，但 graph 的 `nodes[]/leafs[]` 还是空的。调用：

```c
ggml_build_forward_expand(graph, out);
```

内部的 `ggml_visit_parents()` 对 tensor 做 DFS：

1. 首次遇到 tensor 时插入 `visited_hash_set`；已访问的 tensor 直接返回。
2. 按 `order` 递归访问当前 tensor 的 `src[]`。
3. 在本文的推理图中，`op == GGML_OP_NONE` 的 tensor 加入 `leafs[]`。
4. 其余 tensor 加入 `nodes[]`。

tensor 在所有依赖访问完成后才加入数组，因此 `nodes[]` 是后序得到的拓扑顺序：依赖位于使用者之前，新增的计算根通常是最后一个 node。

上述小图可表示为：

```text
w ─┐
   ├─ mul_mat → aw ─┐
a ─┘                ├─ add → sum ─┐
b ──────────────────┘             ├─ mul → out
d ────────────────────────────────┘

leafs: w, a, b, d
nodes: aw, sum, out
```

### `expand` 的增量语义

`ggml_build_forward_expand()` 不会先清空 graph，可以把多个根节点依次加入同一张图：

```c
ggml_build_forward_expand(graph, copy_k);
ggml_build_forward_expand(graph, copy_v);
ggml_build_forward_expand(graph, output);
```

共享依赖因为 visited hash 只会加入一次。这适合在主输出之外加入需要执行的 copy/write 节点。若要从空 graph 重新构建，应先调用 `ggml_graph_clear()`。

### Graph view、复制与查询

- `ggml_graph_view(graph, i0, i1)`：内部工具，浅引用 `nodes[i0:i1)`；结果不带 leaf 和 visited hash。scheduler 用它表示一个 split 的连续 node 区间。
- `ggml_graph_cpy(src, dst)`：复制 node/leaf 指针、遍历顺序及 visited 集合，不复制 tensor metadata 或 data。
- `ggml_graph_dup(ctx, graph)`：在另一个 context 创建 graph object，再调用 `ggml_graph_cpy()` 做相对 tensor 的浅复制。
- `ggml_graph_node()`、`ggml_graph_nodes()`、`ggml_graph_n_nodes()`：取得 node 或 node 数量。
- `ggml_graph_get_tensor()`：按名称在 leaf 和 node 中查找 tensor。
- `ggml_graph_print()`：输出 graph 的 node/leaf 信息。
- `ggml_graph_dump_dot()`：把 graph 写成 Graphviz DOT 文件。
- `ggml_graph_clear()`：清空 node、leaf 数量和 visited 状态，不修改 tensor data。

`ggml_graph_view()` 和 graph 的内部字段定义在 `src/ggml-impl.h`，不属于普通调用者使用的公开接口。

### 当前不可用的声明

`include/ggml.h` 声明了 `ggml_graph_export()` 和 `ggml_graph_import()`，但当前仓库没有对应实现，构建出的 `libggml-base` 也不导出这两个符号，因此当前版本不应调用它们。

## 后端抽象与 CPU 后端

`ggml-backend` 将“在哪里执行”和“数据存在哪里”抽象为两条相关但不同的路径：

```text
Backend Registry -> Device -> Backend
                         \-> Buffer Type -> Buffer -> Tensor data
```

- `ggml_backend_reg_t`：一个后端实现的注册入口，负责报告名称、设备列表和扩展函数。
- `ggml_backend_dev_t`：具体设备的能力与工厂接口，负责创建 backend、提供首选 buffer type，并判断算子和内存类型是否受支持。
- `ggml_backend_t`：一次执行实例。它保存执行状态，用于创建或执行计算计划、提交 graph，以及在需要时同步。
- `ggml_backend_buffer_type_t`（下文简称 `buft`）：一种内存的分配策略，描述名称、对齐、单 buffer 上限、tensor 实际占用大小以及是否为 host 内存。
- `ggml_backend_buffer_t`：一次实际的内存分配，可承载多个 tensor，并实现该内存上的读、写、清零和释放。

这些类型在公共头文件中是不透明指针。应用应使用 `include/ggml-backend.h` 中的函数访问它们，不应依赖 `src/ggml-backend-impl.h` 中的内部字段。特别要注意，tensor 的同步读写属于 **buffer 接口**，不是 backend 的执行接口。

### Registry、静态注册与动态加载

进程内的全局 registry 分别保存 backend registration 和 device 列表：

```c
size_t n_reg = ggml_backend_reg_count();
ggml_backend_reg_t reg = ggml_backend_reg_get(0);

size_t n_dev = ggml_backend_dev_count();
ggml_backend_dev_t dev = ggml_backend_dev_get(0);
```

编译进库的实现会在 registry 首次构造时静态注册。当前 CPU registration 名为 `"CPU"`，报告一个同名 CPU device。registry 保存的是注册对象和设备对象，不是可直接执行的 backend 实例；执行实例仍需由 device 创建。

动态加载有两种入口：

- `ggml_backend_load(path)` 直接加载指定动态库，检查可选的 `ggml_backend_score`，查找 `ggml_backend_init`，并校验 `GGML_BACKEND_API_VERSION` 后注册。
- `ggml_backend_load_all[_from_path]()` 在搜索目录中为同名候选库调用 `ggml_backend_score`，选择得分最高者；没有带变体后缀的候选时，再尝试基础库名。

加载成功后，registry 持有动态库句柄。当前实现还明确注明：若后端资源或线程仍在使用动态库，无法保证安全卸载，因此应用必须先销毁相关 backend、buffer 等资源。

本章后续只讨论 CPU 的具体实现；上述注册机制是所有 backend 共用的基础设施。

### CPU device 与 backend 实例

CPU backend 可以直接创建，也可以经过通用 device 接口创建：

```c
#include "ggml-backend.h"
#include "ggml-cpu.h"

// CPU 专用入口。
ggml_backend_t backend = ggml_backend_cpu_init();

// 等价的通用入口。
ggml_backend_dev_t dev = ggml_backend_dev_by_name("CPU");
ggml_backend_t backend_from_dev =
    dev != NULL ? ggml_backend_dev_init(dev, NULL) : NULL;
```

CPU device 的职责包括：

- 类型为 `GGML_BACKEND_DEVICE_TYPE_CPU`，名称为 `"CPU"`；description 尽量读取平台提供的 CPU 型号。
- 返回默认的 CPU buffer type。
- 从调用方提供的 host 指针创建 buffer。
- 根据算子类型、输入数据类型和输入 buffer 类型实现 `supports_op`。
- 接受 host buffer，以及编译配置可能提供的 CPU extra buffer type。

不要把 device 的 memory 属性当成系统可用内存。当前 `ggml_backend_cpu_device_get_memory()` 尚未实现，`memory_free` 和 `memory_total` 均返回 0。

一个 CPU backend 实例保存以下执行状态：线程数、可选的外部线程池、可复用的临时工作区，以及可选的终止回调。默认线程数为 `GGML_DEFAULT_N_THREADS`。它不拥有模型 tensor 的内存；tensor 数据由相应的 buffer 管理。

### CPU buffer type、buffer 与 tensor 数据访问

`ggml_backend_cpu_buffer_type()` 返回默认 CPU buft：

- 名称为 `"CPU"`，属于 host 内存。
- 使用 `ggml_aligned_malloc()` 分配，并按 `TENSOR_ALIGNMENT` 对齐。
- 未覆盖 `get_max_size`，因此通用默认值为 `SIZE_MAX`。
- 未覆盖 `get_alloc_size`，因此普通 tensor 的默认分配大小为 `ggml_nbytes(tensor)`。

由它创建的 CPU buffer 拥有底层内存，释放 buffer 时会一并释放内存。`ggml_backend_cpu_buffer_from_ptr()` 则把已对齐的外部内存包装成名为 `"CPU_Mapped"` 的 buffer；该 buffer **不拥有指针**，调用方必须保证指针在 buffer 使用期间有效，并在合适的时机自行释放。

同步数据访问直接分派到 tensor 所属的 buffer：

```c
ggml_backend_tensor_set(tensor, src, 0, ggml_nbytes(tensor));
ggml_backend_tensor_get(tensor, dst, 0, ggml_nbytes(tensor));
ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
```

这些函数要求 tensor 已分配、`tensor->buffer` 有效且访问范围不越界。对于 view，代码使用其 `view_src` 的 buffer。CPU buffer 的对应实现最终是 `memcpy` 或 `memset`。

`GGML_BACKEND_BUFFER_USAGE_WEIGHTS`、`COMPUTE` 和 `ANY` 是用途提示，不改变 buffer 的所有权。释放顺序仍由创建者负责。

### 为 context 中的 tensor 分配 CPU buffer

当 context 使用 `no_alloc = true` 创建 tensor 元数据时，可在 tensor 创建完毕后统一分配数据区：

```c
struct ggml_init_params params = {
    .mem_size   = metadata_size,
    .mem_buffer = NULL,
    .no_alloc   = true,
};

struct ggml_context * ctx = ggml_init(params);
// 在 ctx 中创建 tensor……

ggml_backend_buffer_t buffer =
    ggml_backend_alloc_ctx_tensors(ctx, backend);
```

该函数只处理尚未分配数据的 tensor，并使用 backend 的默认 buft。它按对齐和 `get_alloc_size` 计算空间，将 tensor 的 `data`、`buffer` 等字段绑定到实际分配。view 不单独占用数据区，而是由 `ggml_backend_view_init()` 连接到源 tensor。

如果所有 tensor 能放进一个 buffer，函数直接返回该 buffer；只有累计大小超过 buft 的单 buffer 上限时，才分成多个真实 buffer，并返回一个 multi-buffer 包装。这个包装主要用于统一释放、清零和传播 usage，本身没有连续的 base 地址，也不实现普通 tensor 读写。

返回的 buffer 拥有这批数据分配，必须在所有相关 tensor 不再使用后调用 `ggml_backend_buffer_free(buffer)`。更完整的生命周期复用和峰值内存规划由后面的 graph allocator 章节说明。

### CPU graph 执行与 `ggml_cplan`

CPU 的直接执行路径为：

```text
ggml_backend_graph_compute()
        -> CPU backend graph_compute
        -> ggml_graph_plan()
        -> 准备或复用 work_data
        -> ggml_graph_compute()
```

`ggml_graph_plan()` 遍历 graph 节点，为每个算子选择任务数，估算所有节点所需临时内存的最大值，并生成 `ggml_cplan`。plan 中的重要字段是：

- `n_threads`：该 graph 实际使用的线程数，不超过调用方请求值和算子所需最大任务数。
- `work_size` / `work_data`：算子执行所需的临时工作区，不是 tensor 的持久数据区。
- `threadpool`：可选的可复用线程池。
- `abort_callback` / `abort_callback_data`：计算期间的协作式终止检查。

CPU backend 会按需扩充自身的 `work_data`，之后的 graph 计算可以复用该空间。它也实现了显式的 `ggml_backend_graph_plan_create/compute/free`；显式 plan 在创建时一次性分配工作区，适合重复执行结构不变的 graph。当前实现对 `ggml_cgraph` 只是浅拷贝，因此 plan 使用期间必须保证 graph 及其引用的 tensor 元数据仍然有效。

CPU backend 是同步实现：异步 tensor 读写、event 和 `synchronize` 回调均为空。`ggml_backend_graph_compute()` 虽然采用通用的“提交后再同步”包装，但 CPU 的计算已在提交调用中完成，随后的 synchronize 是空操作。

### 线程池、终止回调与 NUMA

最简单的配置只需设置线程数：

```c
ggml_backend_cpu_set_n_threads(backend, n_threads);
enum ggml_status status = ggml_backend_graph_compute(backend, graph);
```

未提供线程池时，每次底层 `ggml_graph_compute()` 都会创建并销毁一次临时线程池。重复推理可显式复用线程池：

```c
struct ggml_threadpool_params tpp =
    ggml_threadpool_params_default(n_threads);
ggml_threadpool_t threadpool = ggml_threadpool_new(&tpp);

ggml_backend_cpu_set_threadpool(backend, threadpool);
enum ggml_status status = ggml_backend_graph_compute(backend, graph);

// 先解除 backend 对线程池的引用，再释放线程池。
ggml_backend_cpu_set_threadpool(backend, NULL);
ggml_threadpool_free(threadpool);
```

`ggml_threadpool_params` 还可配置 CPU mask、调度优先级、轮询强度、严格绑核以及初始暂停状态。调用方创建的线程池仍归调用方所有；`ggml_backend_free()` 不会替调用方释放它。

`ggml_backend_cpu_set_abort_callback()` 可安装终止回调。回调返回 true 时，正在执行的 graph 会尽快以相应状态退出；这是协作式检查，不是立即强制终止线程。

NUMA 主机可在开始计算前调用一次 `ggml_numa_init(strategy)`。`ggml_is_numa()` 表示初始化时是否检测到多个 NUMA node。NUMA 策略会影响线程亲和性和内存访问行为，应在创建工作线程及大规模分配前确定。

### 最小 CPU 执行生命周期

假设 graph 和 tensor buffer 已经构建、分配并填充，CPU 执行端的生命周期如下：

```c
ggml_backend_t backend = ggml_backend_cpu_init();
if (backend == NULL) {
    // 处理初始化失败。
}

ggml_backend_cpu_set_n_threads(backend, n_threads);

enum ggml_status status = ggml_backend_graph_compute(backend, graph);
if (status == GGML_STATUS_SUCCESS) {
    ggml_backend_tensor_get(output, output_data, 0, ggml_nbytes(output));
}

// 先释放依赖 backend/buft 的 tensor buffer 和 context，最后释放 backend。
ggml_backend_buffer_free(buffer);
ggml_free(ctx);
ggml_backend_free(backend);
```

若使用 scheduler，graph 的拆分、buffer 分配和执行提交由 scheduler 统一协调；不要再把 graph 直接提交给单个 backend。scheduler 的完整流程在后文单独说明。

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

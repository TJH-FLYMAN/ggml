# ggml

## 总览与范围

GGML 是一个以 tensor 和计算图为核心的机器学习张量库。本文以当前仓库源码为准，介绍离线推理涉及的核心模块；具体 backend 实现只展开 CPU，不介绍训练、梯度和反向传播。

| 模块 | 核心对象/接口 | 职责 |
| --- | --- | --- |
| tensor 与算子 | `ggml_tensor`、`ggml_type`、`ggml_op` | 描述数据类型、shape、stride、view、算子及依赖关系 |
| context | `ggml_context`、`ggml_object` | 管理 tensor、graph 等对象的 metadata，以及可选的内联 tensor data |
| computation graph | `ggml_cgraph` | 从输出 tensor 回溯依赖并建立前向计算图 |
| backend abstraction 与 CPU | backend、device、buffer type、buffer、`ggml_cplan`、threadpool | 抽象设备、内存和计算接口，并在 CPU backend 中规划和执行 graph |
| allocation | `ggml_tallocr`、`ggml_gallocr` | 分配 tensor，或根据 graph 生命周期规划和复用执行期内存 |
| scheduler | `ggml_backend_sched` | 选择节点 backend、切分子图、创建跨 backend copy、调用 gallocr 分配内存并驱动各 split 执行 |
| quantization | type traits、`ggml_quantize_*` | 定义量化 block 格式、离线转换和 CPU 量化计算能力 |
| GGUF | `gguf_context` | 读取、检查、修改和写出模型 metadata、tensor info 及 tensor data |
| 端到端生命周期 | GGUF、CPU backend、scheduler、graph | 串联权重加载、graph 构建、执行、重复推理和资源释放 |

文档按照“数据表示 → metadata 管理 → graph 构建 → CPU 执行 → 内存规划 → scheduler → 量化与持久化 → 完整部署”的顺序展开。

典型离线推理的数据流为：

```text
GGUF 文件
→ 解析 metadata 和 tensor info
→ 创建并填充权重 tensor
→ 构建前向 ggml_cgraph
→ scheduler 和 allocator 分配计算 tensor
→ CPU backend 执行 graph
→ 读取输出
```

本文完整介绍 scheduler 的通用多 backend 调度机制，但具体 backend 的内部实现只介绍 CPU。最后的端到端章节会把前面各模块串联成一条完整的 CPU 离线推理生命周期。

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

## Tensor allocator 与 graph allocator

`src/ggml-alloc.c` 中有两类用途不同的 tensor allocator：

| 分配器 | 输入 | 分配方式 | 典型用途 |
| --- | --- | --- | --- |
| `ggml_tallocr` | 一个已经存在的 backend buffer | 只向前移动 offset，不回收 | 将一组长期存在的 tensor 顺序放入 buffer，例如 context tensor 的静态分配 |
| `ggml_gallocr` | 一个或多个 buffer type，以及一张 graph | 根据 graph 生命周期规划、回收并复用 offset | graph 的 input、output 和中间结果等计算 tensor |

二者分配的都是 **tensor 数据区**。不要把 gallocr 管理的 compute buffer 与 CPU `ggml_cplan.work_data` 混为一谈：前者保存 graph tensor 的值，后者是 CPU kernel 在一次计算中使用的临时工作区。

### `ggml_tallocr`：单 buffer 线性分配

`ggml_tallocr_new(buffer)` 保存 buffer、base、alignment 和当前 offset。每次调用 `ggml_tallocr_alloc()` 时，它会：

1. 通过 buffer type 的 `get_alloc_size` 取得 tensor 实际分配大小。
2. 按 buffer alignment 补齐大小和地址。
3. 检查剩余容量；容量不足会直接中止。
4. 调用 `ggml_backend_tensor_alloc()`，把 buffer 和 `base + offset` 绑定到 tensor。
5. 将 offset 移到下一段空间。

它没有单 tensor 的 free 或复用操作，也不拥有传入的 buffer。`ggml_backend_alloc_ctx_tensors_from_buft()` 内部会先创建一个或多个实际 buffer，再用 `ggml_tallocr` 将对应范围内尚未分配的 tensor 顺序排入其中。

### `ggml_gallocr` 的角色与所有权

graph allocator 面向整张前向 graph 规划 tensor 数据区。它不只处理“中间 tensor”：只要 tensor 尚无数据且不是 view，graph 中的 leaf、显式 input、普通 node 和 output 都可能由它分配。已经有 `data` 的 tensor 被视为外部分配，不会由 gallocr 接管。

单 buffer 场景可直接使用 CPU buft：

```c
ggml_gallocr_t galloc =
    ggml_gallocr_new(ggml_backend_cpu_buffer_type());
```

`ggml_gallocr_new_n(bufts, n_bufs)` 则创建多个逻辑 buffer slot。`bufts[i]` 表示 slot `i` 使用的内存类型，而不是必然对应一个不同 backend。如果多个 slot 传入的是同一个 buft 指针，它们会共享同一个动态分配器，reserve 后也指向同一个实际 buffer。

gallocr 内部主要保存：

- `bufts[]` 与 `buffers[]`：逻辑 slot 的内存类型及实际 backend buffer。
- `buf_tallocs[]`：每种不同 buft 对应的动态 offset 分配器。
- `hash_set` 与 `hash_values[]`：reserve 期间每个 tensor 的引用数、view 数、buffer id、offset 和占用状态。
- `node_allocs[]` 与 `leaf_allocs[]`：reserve 结果的快照，按 graph 中的位置保存 node 输出、各个 source 及 leaf 的分配信息。

gallocr 拥有其创建的 backend buffer。`ggml_gallocr_free()` 会释放所有不重复的 buffer 和内部规划状态；此后绑定在这些 buffer 上的 tensor 数据地址全部失效。buffer type 本身不由 gallocr 释放。

### `reserve`：模拟生命周期并预留实际 buffer

`ggml_gallocr_reserve()` 或 `ggml_gallocr_reserve_n()` 的工作分成两个阶段：

```text
graph 生命周期模拟 -> 得到每个 tensor 的 (buffer_id, offset, size_max)
                  -> 按各 buft 的峰值大小创建或扩容实际 buffer
```

因此 reserve 虽然不会给传入 graph 的 tensor 写入 `data` 或 `buffer`，但它不只是计算一个数字：它会在 gallocr 内部实际创建或扩容 backend buffer。

生命周期模拟的大致过程如下：

1. 重置 tensor 哈希状态和每个动态 offset 分配器。
2. 先为未预分配的 leaf 规划空间。
3. 遍历 node，统计每个 source 的 `n_children` 和每个 view source 的 `n_views`；带 `GGML_TENSOR_FLAG_INPUT` 的 tensor 提前分配，避免其地址与其他 graph input 重叠。
4. 按 graph 中的 node 顺序模拟执行：确保 source 已分配，再分配当前 node。
5. 当前 node 消耗完 source 后递减引用计数；当 `n_children == 0` 且没有存活 view 时，把 gallocr 拥有的空间归还空闲表。
6. 将最终规划写入 `node_allocs[]` 和 `leaf_allocs[]`，然后按每个动态分配器的 `max_size` 创建或扩容 buffer。

动态 offset 分配器维护按地址排序的 free block。它优先对有限的已回收块做近似 best-fit；都不合适时，才从最后的未使用尾部继续增长。释放时会与相邻 free block 合并。这里的“释放”只是规划阶段将 offset 归还空闲表，不会逐个调用 backend buffer 的 free。

已分配的 buffer 只在新需求更大时扩容，不会因为后续 graph 较小而自动缩小。因此 `ggml_gallocr_get_buffer_size()` 返回当前实际持有的 buffer 大小，它可能大于最近一次 graph 的最低需求。

### 生命周期标志与安全的存储复用

`ggml_set_input(tensor)` 和 `ggml_set_output(tensor)` 会影响 allocator，而不只是给 tensor 添加说明：

- input 会在 graph 模拟的早期分配，保证多个输入拥有互不覆盖的地址。
- output 在其最后一个 graph consumer 结束后仍不会被回收，也不会作为其他 node 的原地复用来源。

对于一组允许原地执行的 op，gallocr 可能让输出直接使用某个 source 的 offset。普通 source 至少需要满足：

- 该存储由当前 gallocr 管理，而不是外部数据。
- source 只剩当前这一个 child，且没有其他存活 view。
- source 与输出的 type、`ne[]` 和 `nb[]` 完全一致。
- source 及其 view source 不是 graph output。

view source 还需要满足更严格的别名和起始地址条件。条件不满足时，allocator 会分配新的 offset。

这种“原地”是 allocator 根据 graph 生命周期进行的 物理存储复用。它不要求调用者使用名称带 `_inplace` 的构图函数，也不改变 graph 的逻辑依赖；能否复用由当前实现的 op allowlist 和上述安全条件共同决定。

### `alloc_graph`：将规划绑定到 tensor

`ggml_gallocr_alloc_graph(galloc, graph)` 使用 reserve 保存的按位置规划，将实际地址绑定到一张 graph：

1. 检查 node/leaf 数量以及待分配 tensor 的大小是否仍能放入保存的 `size_max`。
2. 调用每个 backend buffer 的可选 `reset`；reset 不等于把数据清零。
3. 对普通 tensor 计算 `buffer base + offset`，并调用 `ggml_backend_tensor_alloc()`。
4. view 不申请独立空间，而是在 view source 已有 backend buffer 时调用 `ggml_backend_view_init()`。
5. 已经带有 `data` 的外部 tensor 保持不变。

单 buffer gallocr 在首次使用或检测到数量、大小不匹配时，会自动调用 `ggml_gallocr_reserve()`。多 buffer gallocr 无法自行恢复各 node/leaf 应使用的 buffer id；需要调用方先用正确的映射再次执行 `ggml_gallocr_reserve_n()`，否则 `alloc_graph` 返回 false。

当前实现的自动检查并不会完整比较 graph 的依赖拓扑、view 关系、input/output 标志或 buffer 映射。如果 graph 在这些方面发生变化，即使 node 数量和 tensor 大小相同，也必须显式重新 reserve。用于提前 reserve 的“最大 graph”不仅要覆盖最大 shape，还应代表实际 graph 的 node 顺序、生命周期、view 关系、预分配状态和 input/output 标志。

已经绑定到 gallocr buffer 的 tensor 只能在该 buffer 有效期间使用。重新 reserve 可能扩容并替换旧 buffer，释放 gallocr 则一定释放 buffer；不能继续执行或读取仍指向旧地址的 graph。

### 单 CPU buffer 的典型流程

```c
ggml_gallocr_t galloc =
    ggml_gallocr_new(ggml_backend_cpu_buffer_type());

// 可选：用能代表最大输入和真实生命周期的 graph 提前扩容。
if (!ggml_gallocr_reserve(galloc, measure_graph)) {
    // 处理 buffer 分配失败。
}

// actual_graph 中需要 gallocr 管理的 tensor 此时应尚未分配数据。
if (!ggml_gallocr_alloc_graph(galloc, actual_graph)) {
    // 处理规划失效或 buffer 分配失败。
}

size_t compute_buffer_size =
    ggml_gallocr_get_buffer_size(galloc, 0);
enum ggml_status status =
    ggml_backend_graph_compute(cpu_backend, actual_graph);

// actual_graph 的计算 tensor 数据在这里失效。
ggml_gallocr_free(galloc);
```

提前 reserve 是可选优化。单 buffer 场景可直接调用 `alloc_graph`，由它按需 reserve；显式使用代表性 graph 可以避免输入 shape 在预期范围内变化时反复扩容。

### 多 buffer 与 scheduler 的边界

直接使用 `ggml_gallocr_reserve_n()` 时：

- `node_buffer_ids[i]` 指定 `graph->nodes[i]` 的逻辑 buffer slot。
- `leaf_buffer_ids[i]` 指定 `graph->leafs[i]` 的逻辑 buffer slot。
- 映射、graph 结构或所需大小变化后，调用方负责重新 reserve。

如果多个 slot 因相同 buft 而共享实际 buffer，`ggml_gallocr_get_buffer_size()` 只在该实际 buffer 第一次出现的 id 上返回大小，后续重复 id 返回 0，避免汇总时重复计数。

scheduler 负责确定 node/leaf 的 buffer id、插入必要的 copy tensor 并组织 split graph，再把这些映射交给 gallocr；gallocr 只根据给定 graph 和映射决定物理 offset。使用 scheduler 时，应用通常不应再单独为同一 graph 调用 gallocr。后文会继续说明 scheduler 如何驱动 `reserve_n()` 和 `alloc_graph()`。

## Backend scheduler

`ggml_backend_sched` 位于 backend 执行接口与 graph allocator 之间，负责：

```text
用户 graph
    -> 为 tensor/node 选择 backend
    -> 按 backend 切分 split
    -> 创建必要的 copy tensor
    -> 调用 gallocr 规划各 buffer
    -> 按 split 顺序提交执行
```

scheduler 的接口支持多个 backend，但本节只具体使用 CPU。即使 scheduler 中只有一个 CPU backend，它仍可统一管理 graph tensor 的计算 buffer、提前 reserve、graph 分配、重复执行和 reset；只是所有 node 都会落在 CPU 上，通常只有一个 split，也不会发生跨 backend copy。

如果应用只需要执行一张简单的 CPU graph，可以直接组合 `ggml_gallocr` 和 `ggml_backend_graph_compute()`。如果希望统一管理最大 graph 预留、输入分配、重复执行及未来的 backend 调度，使用 scheduler 更方便。

### 创建参数与对象所有权

单 CPU scheduler 可以这样创建：

```c
ggml_backend_t cpu_backend = ggml_backend_cpu_init();

ggml_backend_t backends[] = {
    cpu_backend,
};

ggml_backend_sched_t sched = ggml_backend_sched_new(
    backends,
    NULL,                    // 使用各 backend 的默认 buft
    1,
    GGML_DEFAULT_GRAPH_SIZE,
    false);                  // CPU-only 不启用 pipeline copies
```

`ggml_backend_sched_new()` 的参数含义如下：

- `backends[]`：执行 backend 数组，索引越小优先级越高。
- `n_backends`：backend 数量，必须大于 0 且不超过 `GGML_SCHED_MAX_BACKENDS`。
- `backends[n_backends - 1]`：代码要求最后一个 backend 的 device 类型必须是 CPU。
- `bufts[]`：可选的计算 buffer type 数组；传入 `NULL` 时使用各 backend 的默认 buft。每个 backend 必须支持对应 buft。
- `graph_size`：scheduler 内部 hash、node/leaf 映射及临时 graph 的容量依据。必须覆盖实际 graph 的 node 和 leaf 数量，否则会触发断言。
- `parallel`：是否启用多个 pipeline copy slot。

scheduler 拥有：

- 内部 gallocr 及其计算 buffer。
- backend/slot 映射表。
- copy tensor 和依赖 tensor 的元数据 context。
- split 数组、内部 allocation graph 和可选 event。

scheduler 不拥有：

- 传入的 backend 实例。
- 用户创建的 graph 和 tensor 元数据 context。
- 已经预分配的权重或输入 buffer。

因此释放顺序应为：先确保执行完成，再释放 scheduler，最后释放 backend。`ggml_backend_sched_free()` 不会替调用方释放 backend。

### Tensor 到 backend 的映射

scheduler 内部通过 tensor 指针哈希记录：

- `hv_tensor_backend_ids`：每个 tensor 当前分配到的 backend id。
- `hv_tensor_copies`：每个 tensor 在不同 backend 和 copy slot 上的副本。
- `node_backend_ids` / `leaf_backend_ids`：交给 gallocr 的逻辑 buffer id。
- `prev_node_backend_ids` / `prev_leaf_backend_ids`：上一轮分配，用于判断物理 buffer 规划是否需要更新。

应用可选地手动指定 tensor：

```c
ggml_backend_sched_set_tensor_backend(
    sched, tensor, cpu_backend);

ggml_backend_t assigned =
    ggml_backend_sched_get_tensor_backend(sched, tensor);
```

手动指定的 backend 必须属于当前 scheduler。修改 backend 指派或准备一张新 graph 前，必须先 reset scheduler。

通常不需要逐个手动指定。scheduler 会综合以下信息自动决定：

- backend 是否支持该 op。
- backend 是否能直接访问 tensor 当前使用的 buft。
- tensor 是否已经预分配。
- source buffer 是否标记为 `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`。
- 用户手动指定的 backend。
- backend 数组中的优先级。

### Backend 分配的五个阶段

`ggml_backend_sched_split_graph()` 大致通过五个阶段完成 node 分配和 graph 切分。

#### 第一阶段：根据已有位置确定初始 backend

scheduler 首先处理 leaf 和 node：

- 用户通过 `ggml_backend_sched_set_tensor_backend()` 设置的结果不会被覆盖。
- 已有 backend buffer 的 tensor 优先放到能够直接访问该 buft、同时支持对应 op 的最高优先级 backend。
- view 会参考 `view_src` 的 buffer 位置。
- 带 `GGML_TENSOR_FLAG_INPUT` 的 graph input 默认分配给最后一个 backend，即强制要求存在的 CPU fallback。
- 如果某个 source buffer 标记为 `WEIGHTS`，使用该权重的 op 会优先选择能直接使用这块 buffer 的 backend。

如果 tensor 已经预分配，但 scheduler 中没有任何 backend 能使用其 buft 执行对应 op，当前实现会中止，而不是自动移动这个预分配 tensor。

#### 第二阶段：向相邻 node 扩散已有分配

scheduler 会沿 node 顺序向前和向后传播 backend 分配：

1. 先传播除最后一个 CPU fallback 以外的已有分配。
2. 只给该 backend 支持的 op 赋值。
3. 再传播剩余 backend 的分配，填补仍未决定的相邻 node。

这样可尽量让连续、兼容的 node 留在同一 backend，减少 split 和 tensor copy。

在只有 CPU 的场景中，前两次“非 fallback backend”传播不会产生分配；后续阶段最终会把受支持的 node 分配给 CPU。

#### 第三阶段：处理未分配 node 与优先级提升

对于仍未分配的 node，scheduler 会遍历 backend：

- 先要求 backend 支持当前 op。
- 统计该 backend 能直接使用多少个 source。
- 选择兼容 source 数量最多的 backend。
- 数量相同时，较低索引的 backend 因遍历顺序获得优先级。

对于已经分配的 node，如果更高优先级 backend 使用相同 buft、支持当前 op，并能使用所有 source，scheduler 可以将 node 提升到该 backend。

#### 第四阶段：补齐 source 和 view 的 backend

完成 node 分配后：

- view 跟随 `view_src`。
- 尚未分配的普通 source 通常继承使用它的当前 node 的 backend。
- 最终进入 split 阶段前，所有参与计算的 node 和 source 都应有有效 backend id。

#### 第五阶段：切分 split 并创建 copy tensor

`ggml_backend_sched_split` 表示一段连续的 graph node，包含：

- `backend_id`：负责执行该 split 的 backend。
- `i_start` / `i_end`：原始 graph node 数组中的范围。
- `inputs[]`：执行前需要复制到该 backend 的 tensor。
- `graph`：通过 `ggml_graph_view()` 创建的原 graph 视图。

以下情况会开始新的 split：

- 当前 node 的 backend 与前一个 split 不同。
- 当前 split 已经有跨 backend input，又遇到位于不兼容 buffer 上的权重。
- 当前 split 的跨 backend input 数量达到上限，又出现新的不兼容 input。

backend id 不同不一定需要复制。如果目标 backend 能直接访问 source 的 buft，原 tensor 可以直接使用。只有目标 backend 不能访问该 buffer 时，scheduler 才创建 copy tensor。

copy tensor 由 `ggml_dup_tensor_layout()` 创建，只复制 type、shape 和 stride 等布局，不复制数据。实际数据在执行 split 前通过 backend copy 接口传输。

创建 copy tensor 后，scheduler 会把相关 node 的 `src[j]` 改为当前 copy slot 对应的 tensor。因此 split 过程可能修改用户 graph 中 node 的 source 指针；reset 后不能继续把旧 graph 当作一张未分配的新 graph 使用。

### 内部 allocation graph

scheduler 还会构造一张内部 `sched->graph`，供 gallocr 规划内存。它与用户原始 graph 不完全相同：

1. 对每个跨 backend input，加入一个依赖 view，防止 source 在 copy 完成前被 gallocr 回收。
2. 加入 copy tensor，使其在 split 开始前获得目标 buffer 地址。
3. 加入原 graph 中属于各 split 的 node。
4. pipeline 模式下，将多个 copy slot 的输入副本加入 leaf。
5. 最后加入原 graph 的 leaf。

`split->graph` 是原 graph 某段 node 的 view；`sched->graph` 则是为了内存规划而添加了依赖和 copy tensor 的 graph。

scheduler 负责决定“由谁执行、哪里需要复制”；gallocr 负责决定这些 tensor 在各 buft 对应 buffer 中的 offset。

### Reserve 与 graph 分配

`sched_reserve` 的流程是：

```text
measure graph
    -> split_graph
    -> synchronize all backends
    -> ggml_gallocr_reserve_n
    -> scheduler reset
```

调用方式：

```c
if (!ggml_backend_sched_reserve(sched, measure_graph)) {
    // 处理计算 buffer 分配失败。
}
```

reserve 是可选优化。measure graph 应代表最大 shape，以及实际 node 顺序、view、input/output 标志和 backend 分配。reserve 完成后 scheduler 会自动 reset，但 gallocr 已申请的物理 buffer 会保留。

实际 graph 分配通过：

```c
if (!ggml_backend_sched_alloc_graph(sched, graph)) {
    // 处理分配失败。
}
```

其流程为：

1. 再次对实际 graph 分配 backend 并切 split。
2. 尝试使用 gallocr 已保存的规划绑定 tensor。
3. 如果 node/leaf 对应的 buft 发生变化，或 gallocr 判断原规划不足，则先同步所有 backend。
4. 调用 `ggml_gallocr_reserve_n()` 重新规划。
5. 再次调用 `ggml_gallocr_alloc_graph()` 绑定 tensor。

代码比较 backend id 时还会比较 buft。如果 backend id 变化但两个位置使用同一个 buft，不会仅因为 backend id 不同而强制重新规划物理 buffer。

`ggml_backend_sched_get_buffer_size()` 返回某个 backend slot 对应的 gallocr buffer 大小。多个 slot 共用同一 buft 和实际 buffer 时，后续重复 slot 可能返回 0，以避免重复统计。

### Graph 执行

同步入口为：

```c
enum ggml_status status =
    ggml_backend_sched_graph_compute(sched, graph);
```

它内部先调用异步入口，再同步 scheduler 中的所有 backend：

```text
graph_compute
    -> graph_compute_async
    -> synchronize all backends
```

`ggml_backend_sched_graph_compute_async()` 会：

1. 在需要时自动调用 `ggml_backend_sched_alloc_graph()`。
2. 依次处理各 split。
3. 在 split 执行前传输不兼容的输入。
4. 调用该 split backend 的 `ggml_backend_graph_compute_async()`。
5. 轮换 pipeline copy slot。

虽然接口名包含 async，但 CPU backend 的 graph compute 本身是同步实现。因此 CPU-only scheduler 中，每个 split 在提交调用返回前已经计算完成，显式 synchronize 是空操作。

同一张已经分配的 graph 可以重复执行：

```c
for (int i = 0; i < n_runs; ++i) {
    ggml_backend_tensor_set(
        input, input_data[i], 0, ggml_nbytes(input));

    enum ggml_status status =
        ggml_backend_sched_graph_compute(sched, graph);
}
```

重复执行时 scheduler 不会重新 split 或重新分配。传给 compute 的必须仍是已经分配的同一张 graph；若换成新 graph，必须先 reset。

### 跨 backend copy

对于 split input，scheduler 区分两种情况：

- 用户 input：先确保目标 copy slot 不再使用，然后立即执行 copy，避免调用方过早改写输入数据。
- graph 中间结果：等待目标 backend 可安全覆盖 copy slot，再尝试异步 copy；不支持时同步相关 backend，并回退到普通 tensor copy。

这一机制是 scheduler 的通用行为。CPU-only 场景中所有 tensor 通常使用同一个 CPU buft，不需要创建跨 backend copy。

### Pipeline copy slot

`parallel=false` 时：

```text
n_copies = 1
```

`parallel=true` 时：

```text
n_copies = GGML_SCHED_MAX_COPIES = 4
```

scheduler 会为跨 backend input 创建多个副本，并尝试为每个 backend/copy slot 创建 event。执行完成后：

```c
cur_copy = (cur_copy + 1) % n_copies;
```

如果 device 不支持 event，event 指针为 `NULL`，scheduler 会回退到 backend synchronize。

当前 CPU device 不支持异步操作和 event，CPU graph compute 也是同步执行。因此 CPU-only 场景启用 `parallel=true` 不会带来 pipeline overlap，反而会增加 copy tensor 和计算 buffer 占用，应使用 `parallel=false`。

### Eval callback

scheduler 可在执行 node 期间观察中间结果：

```c
ggml_backend_sched_set_eval_callback(
    sched, callback, user_data);
```

callback 分为两个阶段：

- `ask == true`：scheduler 询问是否需要观察当前 node。它可将连续且不需要观察的 node 合并为一个 graph view 一次执行。
- `ask == false`：目标 node 已经执行完成并同步，callback 可以读取结果。

启用 callback 会切小 split 的执行范围，并在观察节点前同步 backend，因此可能明显影响性能。

公共头文件说明 callback 在第二阶段返回 false 可以取消 graph。但当前实现只跳出当前 split 的 node 遍历，没有返回取消状态，之后的 split 仍可能继续执行。这是当前版本的实现限制，不应依赖它完成可靠的全 graph 取消。

### Reset 与 tensor 失效

准备新 graph 或修改手动 backend 指派前必须调用：

```c
ggml_backend_sched_reset(sched);
```

reset 会：

- 清空 tensor/backend 哈希关系。
- 清空 copy tensor 映射。
- 将 scheduler 标记为尚未分配 graph。
- 保留 gallocr 已经申请的物理 buffer，以便后续 graph 复用容量。

reset 不会把旧 graph tensor 的 `data` 清成 `NULL`，但这些地址在逻辑上已经失效。正确流程是：

1. 确保上一轮执行已经同步完成。
2. reset scheduler。
3. 丢弃旧 graph 及其 tensor 元数据。
4. 构建一张新的、尚未分配计算 tensor 的 graph。
5. 重新调用 `alloc_graph()` 或让第一次 compute 自动分配。

不能 reset 后继续使用旧 tensor，也不能直接把一张新 graph 传给仍处于 `is_alloc` 状态的 scheduler。

### 单 CPU scheduler 的完整流程

```c
ggml_backend_t cpu_backend = ggml_backend_cpu_init();
if (cpu_backend == NULL) {
    // 处理初始化失败。
}

ggml_backend_t backends[] = {
    cpu_backend,
};

ggml_backend_sched_t sched = ggml_backend_sched_new(
    backends,
    NULL,
    1,
    GGML_DEFAULT_GRAPH_SIZE,
    false);

// 可选：使用代表最大输入的 measure graph 提前 reserve。
// ggml_backend_sched_reserve(sched, measure_graph);

// 构建尚未分配计算 tensor 的实际 graph。
struct ggml_cgraph * graph = build_graph(graph_ctx);

if (!ggml_backend_sched_alloc_graph(sched, graph)) {
    // 处理 graph buffer 分配失败。
}

// graph input 必须在 alloc_graph 之后写入。
ggml_backend_tensor_set(
    input, input_data, 0, ggml_nbytes(input));

enum ggml_status status =
    ggml_backend_sched_graph_compute(sched, graph);

if (status == GGML_STATUS_SUCCESS) {
    ggml_backend_tensor_get(
        output, output_data, 0, ggml_nbytes(output));
}

// graph_compute 是同步入口；使用异步入口时需在这里显式同步。
ggml_backend_sched_synchronize(sched);

// reset 后旧 graph tensor 地址失效，随后丢弃元数据。
ggml_backend_sched_reset(sched);
ggml_free(graph_ctx);

// scheduler 释放内部 gallocr 和计算 buffer，但不释放 CPU backend。
ggml_backend_sched_free(sched);
ggml_backend_free(cpu_backend);
```

调试 backend 分配和 split 时，可在创建 scheduler 前设置环境变量 `GGML_SCHED_DEBUG`。非零值打印 split 信息，较高值还会输出逐 node 的 backend 分配及原因。

## 量化、block 布局与 CPU type traits

在 GGML 中，量化类型不是“每个元素占固定字节数”的普通标量类型，而是一种 block 编码格式：若干逻辑元素共享 scale、min、查找表索引等元数据，并被压缩到一个存储 block 中。

离线推理中的量化可以分成两个阶段：

1. 模型转换阶段：把 F32 权重转换成指定量化格式，通常写入 GGUF。
2. 模型执行阶段：CPU kernel 直接读取量化权重，并根据对应的 CPU type traits 选择转换函数和 dot kernel。

### block 大小与实际存储量

通用 type traits 中的两个大小必须区分：

- `ggml_blck_size(type)`：一个存储 block 表示多少个逻辑元素。
- `ggml_type_size(type)`：一个完整存储 block 占多少字节。

因此，一行数据的大小不是简单的“元素数 × `ggml_type_size()`”，而是：

```text
row_bytes = ggml_type_size(type) * ne0 / ggml_blck_size(type)
```

对应接口是：

```c
size_t row_bytes = ggml_row_size(type, ne0);
```

`ggml_row_size()` 要求 `ne0` 能被该类型的 block size 整除，否则会触发断言。

一个量化格式包含 scale、min 等额外信息，所以格式名称中的位数不等于最终平均存储位数。例如：

- `Q4_0` 的一个 block 表示 32 个值，由一个 F16 scale 和 16 字节量化值组成，共 18 字节，平均为 `18*8/32 = 4.5` bit/value。
- `Q4_1` 在量化值之外还保存 scale 和 min，共 20 字节，平均为 `20*8/32 = 5` bit/value。

可以用下面的公式计算某个格式包含 block 元数据之后的平均存储位数：

```text
effective_bits_per_value =
    8 * ggml_type_size(type) / ggml_blck_size(type)
```

### 主要量化格式族

具体 block 结构定义在 `src/ggml-common.h`，量化和反量化实现在 `src/ggml-quants.c`。

| 格式族 | 典型类型 | block 特点 |
| --- | --- | --- |
| 经典 Q 格式 | `Q4_0`、`Q4_1`、`Q5_0`、`Q5_1`、`Q8_0`、`Q8_1` | 通常每个 block 表示 32 个值；不同格式保存的 scale、min 和高位数据不同 |
| K-quant | `Q2_K`、`Q3_K`、`Q4_K`、`Q5_K`、`Q6_K` | 使用 256 个值的 super-block，并在其中保存分组 scale、min 或高位数据 |
| IQ | `IQ1_*`、`IQ2_*`、`IQ3_*`、`IQ4_*` | 多数使用 256 个值的 super-block，并使用非线性 codebook、grid 或索引结构；`IQ4_NL` 的 block size 为 32 |
| TQ | `TQ1_0`、`TQ2_0` | 以 256 个值为一个 block 的 ternary 量化格式 |
| CPU 中间格式 | `Q8_1`、`Q8_K` | 主要作为某些 dot kernel 的 activation companion；其中 `Q8_K` 的源码注释明确说明其用于中间量化和 dot product，而不是普通持久化权重格式 |

这些格式不能只按照名称中的位数解释。每个格式都有独立的 block 字段布局、量化范围和 CPU kernel，不能把一种格式的裸数据按另一种格式读取。

旧的 `Q4_0_4_4`、`Q4_0_4_8`、`Q4_0_8_8` 以及对应的 `IQ4_NL` interleave 类型已经从通用存储类型中移除。当前源码为这些枚举槽位设置了零 block size，并提示使用普通 `Q4_0` 或 `IQ4_NL` 配合运行时 repacking；它们不能再作为有效的持久化 tensor 类型使用。

### 通用 `ggml_type_traits`

`ggml_get_type_traits(type)` 返回该存储类型的通用描述：

```c
struct ggml_type_traits {
    const char      * type_name;
    int64_t           blck_size;
    int64_t           blck_size_interleave;
    size_t            type_size;
    bool              is_quantized;
    ggml_to_float_t   to_float;
    ggml_from_float_t from_float_ref;
};
```

各字段的职责为：

| 字段 | 含义 |
| --- | --- |
| `type_name` | 类型名称 |
| `blck_size` | 每个存储 block 表示的逻辑元素数 |
| `blck_size_interleave` | block 内交错信息；当前通用类型表多数为 0，CPU 的特定布局优化主要通过运行时 repacking 实现 |
| `type_size` | 一个完整 block 的字节数 |
| `is_quantized` | 是否为量化类型 |
| `to_float` | 把当前格式转换为 F32 的参考入口 |
| `from_float_ref` | 从 F32 转换为当前格式的参考入口 |

转换函数指针不保证非空。例如，`Q8_K` 没有通用 `to_float` 和 `from_float_ref`，部分 IQ 类型也没有 `from_float_ref`。调用方必须先检查函数指针，不能假设所有 `ggml_type` 都支持双向通用转换。

`src/ggml-quants.h` 中的 `quantize_row_*`、`quantize_row_*_ref` 和 `dequantize_row_*` 是量化实现及 CPU backend 使用的行级函数。虽然其中部分函数带有导出标记，但应用层进行离线模型转换时，更合适的统一入口是 `ggml_quantize_chunk()`。

### 使用 `ggml_quantize_chunk()` 离线转换

`ggml_quantize_chunk()` 把连续的 F32 源数据转换为目标存储格式：

```c
size_t ggml_quantize_chunk(
        enum ggml_type   type,
           const float * src,
                  void * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           const float * imatrix);
```

典型用法如下：

```c
enum ggml_type type = GGML_TYPE_Q4_0;

int64_t nrows       = ...;
int64_t n_per_row   = ...;

size_t row_size = ggml_row_size(type, n_per_row);
size_t dst_size = nrows * row_size;

void * dst = ...; // 至少提供 dst_size 字节

size_t written = ggml_quantize_chunk(
    type,
    src_f32,
    dst,
    0,
    nrows,
    n_per_row,
    NULL);

GGML_ASSERT(written == dst_size);
```

上面的示例使用 `start == 0`，所以 `dst_size = nrows * row_size` 足够。使用非零 `start` 时，源和目标 buffer 都必须包含起始偏移之前的空间。

调用时需要满足以下条件：

1. `type` 必须是当前 `ggml_quantize_chunk()` 支持的目标类型。
2. `src` 和 `dst` 必须指向有效 buffer。
3. `start >= 0`、`nrows >= 0` 且 `n_per_row > 0`。
4. `n_per_row` 必须能被目标类型的 block size 整除。
5. `start` 是源 F32 数组中的元素偏移，而不是字节偏移。
6. `start` 必须同时满足 `start % n_per_row == 0` 和 `start % blck_size == 0`，即本次转换从所声明的行边界开始。
7. 调用方必须预先检查 `nrows * n_per_row`、`start + nrows * n_per_row`、`start_row * row_size` 和相关 buffer 大小计算不会发生整数溢出。
8. 从原始 `src` 基地址计算，源 buffer 至少需要包含：

```text
start + nrows * n_per_row
```

个 F32 元素。

9. 从原始 `dst` 基地址计算，目标 buffer 至少需要包含：

```text
(start / n_per_row + nrows)
    * ggml_row_size(type, n_per_row)
```

字节。函数不会替调用者分配或扩容目标 buffer。

10. 当 `imatrix != NULL` 且目标格式使用 importance weights 时，它至少需要包含 `n_per_row` 个 F32 权重；当前实现会在本次调用处理的每一行中重复使用这组权重，不会根据 `start` 自动移动指针。
11. 返回值是本次实际写入的目标格式字节数，源码会检查它等于：

```text
nrows * ggml_row_size(type, n_per_row)
```

目标写入位置按照下面的方式计算：

```text
start_row = start / n_per_row
dst_offset = start_row * ggml_row_size(type, n_per_row)
```

因此，可以用多个不重叠 chunk 填充同一个目标 buffer，但每个 chunk 都必须使用相同的逻辑布局，并遵守它所声明的行宽、行边界和完整 buffer 容量。

`ggml_quantize_chunk()` 本身没有完整检查参数的正负值、乘加溢出以及源/目标 buffer 容量。满足这些条件是调用方的责任。

当前 `ggml_quantize_chunk()` 支持：

- `Q4_0`、`Q4_1`、`Q5_0`、`Q5_1`、`Q8_0`；
- `Q2_K`、`Q3_K`、`Q4_K`、`Q5_K`、`Q6_K`；
- `IQ2_XXS`、`IQ2_XS`、`IQ2_S`、`IQ3_XXS`、`IQ3_S`、`IQ1_S`、`IQ1_M`、`IQ4_NL`、`IQ4_XS`；
- `TQ1_0`、`TQ2_0`；
- F32 到 `F16`、`BF16` 和 `F32` 的转换或复制。

`Q8_1`、`Q8_K` 以及已经移除的 interleave 类型不是该接口支持的目标类型；传入未支持的类型会进入断言失败路径。

### importance matrix

`imatrix` 是离线量化时可选的外部 importance weights，用于让部分量化算法在误差计算中提高某些位置的重要程度。本文只讨论如何把它传给离线转换接口，不讨论这些权重如何生成。

当前实现把 `imatrix` 解释为一行中各位置的 importance weights，并在本次调用处理的各行之间重复使用；它不会根据 `start` 自动移动 `imatrix` 指针。

可以用下面的接口判断目标格式是否强制要求 importance matrix：

```c
bool required = ggml_quantize_requires_imatrix(type);
```

当前源码只对以下类型返回 `true`：

- `GGML_TYPE_IQ2_XXS`
- `GGML_TYPE_IQ2_XS`
- `GGML_TYPE_IQ1_S`

`IQ1_M` 在该判断中的代码目前被注释掉，因此当前实现不会把它判定为强制需要 `imatrix`。对强制需要的类型传入 `NULL` 会触发断言。对其他部分格式，`imatrix` 可以为空，也可以作为可选权重参与量化；还有一些格式会直接忽略该参数。

### 量化查找表的生命周期

部分 IQ 格式需要初始化运行时查找表。接口为：

```c
ggml_quantize_init(type);
ggml_quantize_free();
```

`ggml_quantize_chunk()` 会自动调用 `ggml_quantize_init(type)`，所以普通调用者不需要在每个 chunk 前手动初始化。重复初始化同一种类型不会重复创建已经存在的表。

头文件把 `ggml_quantize_init()` 和 `ggml_quantize_free()` 声明为线程安全，源码也使用全局临界区串行化初始化和释放。但释放操作不应与正在使用这些查找表的量化操作并发进行；通常应在全部离线转换完成之后统一调用：

```c
ggml_quantize_free();
```

当前源码还有一处实现与头文件注释不完全一致：

- `ggml_quantize_init()` 可以为 `IQ2_S` 初始化独立的 IQ2 表，也可以为 `IQ3_S` 初始化 grid size 为 512 的 IQ3 表。
- `ggml_quantize_free()` 当前释放了 `IQ2_XXS`、`IQ2_XS`、`IQ1_S/IQ1_M` 共享表和 grid size 为 256 的 IQ3 表，但没有释放 `IQ2_S` 的独立表和 `IQ3_S` 使用的 512 grid。

因此，当前版本的 `ggml_quantize_free()` 没有完全兑现头文件中“释放 `ggml_quantize_init()` 分配的所有内存”的注释。这是当前源码的实现限制；长生命周期进程如果反复初始化和释放这些格式，需要特别注意。

### CPU `ggml_type_traits_cpu`

通用 `ggml_type_traits` 描述存储格式和参考转换，CPU backend 还维护一张计算能力表：

```c
struct ggml_type_traits_cpu {
    ggml_from_float_t from_float;
    ggml_vec_dot_t    vec_dot;
    enum ggml_type    vec_dot_type;
    int64_t           nrows;
};
```

可以通过下面的接口取得：

```c
const struct ggml_type_traits_cpu *
ggml_get_type_traits_cpu(enum ggml_type type);
```

该结构在 `ggml-cpu.h` 中被归类为供测试和 benchmark 使用的 internal types/functions。理解它有助于阅读 CPU kernel，但应用层不应把它当成稳定的模型转换接口。

字段含义为：

| 字段 | 含义 |
| --- | --- |
| `from_float` | CPU 优化的 F32 到该类型的转换函数；不保证存在 |
| `vec_dot` | 该类型作为左侧数据时使用的 CPU dot kernel |
| `vec_dot_type` | dot kernel 要求右侧输入转换成的 companion 类型 |
| `nrows` | kernel 一次协同处理的行数，可能随类型和编译目标变化 |

常见的权重类型与 activation companion 关系如下：

| 权重类型 | `vec_dot_type` |
| --- | --- |
| `Q4_0`、`Q5_0`、`Q8_0`、`IQ4_NL` | `Q8_0` |
| `Q4_1`、`Q5_1` | `Q8_1` |
| `Q2_K`～`Q6_K`、大多数 IQ 类型、`IQ4_XS`、`TQ1_0`、`TQ2_0` | `Q8_K` |

在常见的 CPU 矩阵乘路径中，CPU backend 会按照量化权重类型取得 `vec_dot` 和 `vec_dot_type`。如果 activation 还不是 kernel 要求的 companion 类型，执行计划会使用 CPU work buffer 中的临时空间，通过 companion 类型的 `from_float` 将 activation 转换后再调用 dot kernel。

因此，量化推理通常不是：

```text
量化权重 -> 全部反量化为 F32 -> F32 矩阵乘
```

而更接近：

```text
量化权重 block
        +
activation -> 临时 Q8 companion
        ↓
对应的 quantized-weight × Q8 dot kernel
```

这使 CPU kernel 可以直接消费量化权重，避免在每次推理前把全部权重展开成 F32。具体是否走该路径仍取决于算子、数据布局、CPU 特性和对应 kernel 是否可用，不能把它推广为所有 CPU 算子的统一行为。

部分 IQ 类型的 CPU `from_float` 为空，因为其量化过程需要额外初始化或 importance matrix。离线生成模型权重时，应使用 `ggml_quantize_chunk()`，不能因为某种类型存在 `vec_dot` 就假设 CPU traits 同时提供该类型的直接量化函数。

### 量化数据与 GGUF 的边界

GGUF tensor info 保存 tensor 的 `ggml_type`、shape 和数据偏移，tensor data 区保存对应格式的原始 block 字节。GGUF loader 不会因为 tensor 是量化类型就自动把它展开为 F32。

因此，加载量化权重时需要保持以下一致性：

- GGUF 中记录的 tensor type 必须与实际 block 字节格式一致。
- tensor 第 0 维必须满足该类型的 block size 约束。
- 行大小应通过 `ggml_row_size()` 计算。
- tensor 总字节数、GGUF offset 和 backend buffer 中的目标范围必须匹配。
- 不能把量化数据当成按逻辑元素连续排列的标量数组访问。

量化格式负责“数据如何编码”，CPU type traits 和 kernel 负责“这些编码如何参与计算”，GGUF 则负责“类型信息和原始 block 如何持久化”。这三个层次需要保持一致。

## GGUF：模型元数据与 tensor 数据持久化

GGUF 是 GGML 使用的二进制模型容器。它负责持久化：

- 模型和 tokenizer 等 key-value 元数据；
- tensor 的名称、shape、`ggml_type` 和数据偏移；
- tensor 的原始连续存储字节，包括量化 block。

GGUF 不描述 GGML computation graph，也不会自动创建模型的前向算子。加载 GGUF 后，应用仍需根据模型架构自行取得权重 tensor、构建 graph、分配执行期内存并调用 backend。

### 文件结构

当前实现读写的 GGUF 文件按以下顺序排列：

```text
[ magic ]
[ version, n_tensors, n_kv ]
[ key-value metadata ]
[ tensor info ]
[ padding to alignment ]
[ tensor data blob ]
```

各部分含义如下：

| 部分 | 内容 |
| --- | --- |
| magic | 固定的 4 个 ASCII 字节 `GGUF` |
| version | `uint32_t` 版本号 |
| n_tensors | 文件中的 tensor 数量，当前实现按 `int64_t` 读写 |
| n_kv | metadata key-value 数量，当前实现按 `int64_t` 读写 |
| metadata | key、value type 和对应的标量或数组数据 |
| tensor info | tensor 名称、维数、各维长度、`ggml_type` 和相对数据偏移 |
| padding | 把 tensor data 区起点补齐到 `alignment` |
| tensor data | 各 tensor 的原始连续数据以及 tensor 之间的 alignment padding |

magic 只是文件类型标识，不是字节序标识。当前 `src/gguf.cpp` 直接读写整数和浮点值的内存表示，没有执行字节序转换。

`GGUF_VERSION` 当前为 3，新建 context 写出 v3。读取代码显式拒绝 v1 和高于 v3 的版本，因此正常使用的是 v2 或 v3；当前实现没有单独拒绝版本号 0，这是读取校验的一处边界限制。

GGUF string 的编码为：

```text
uint64_t length
byte[length] content
```

内容后面不保存 C 字符串的 `\0`。`gguf_type` 和 `ggml_type` 枚举按 `int32_t` 写入，bool 按 `int8_t` 写入。

### Metadata 类型与访问

GGUF metadata 的类型由 `enum gguf_type` 描述，包括：

- 无符号和有符号整数：U8、I8、U16、I16、U32、I32、U64、I64；
- F32 和 F64；
- bool；
- string；
- array。

array 记录一种元素类型和元素数量，不支持嵌套 array。string array 的每个元素分别使用 GGUF string 编码。

访问 metadata 时，通常先查找 key，再检查类型：

```c
int64_t key_id = gguf_find_key(ctx_gguf, "general.architecture");
if (key_id < 0) {
    // key 不存在
}

if (gguf_get_kv_type(ctx_gguf, key_id) != GGUF_TYPE_STRING) {
    // 类型不符合预期
}

const char * architecture = gguf_get_val_str(ctx_gguf, key_id);
```

主要查询接口为：

| 接口 | 作用 |
| --- | --- |
| `gguf_get_n_kv()` | 返回 key-value 数量 |
| `gguf_find_key()` | 按名称查找 key，未找到时返回 -1 |
| `gguf_get_key()` | 按 id 返回 key 名称 |
| `gguf_get_kv_type()` | 返回顶层类型；array 返回 `GGUF_TYPE_ARRAY` |
| `gguf_get_arr_type()` | 返回 array 的元素类型 |
| `gguf_get_arr_n()` | 返回 array 元素数量 |
| `gguf_get_val_*()` | 读取对应类型的标量 |
| `gguf_get_arr_data()` | 返回非 string array 的原始元素地址 |
| `gguf_get_arr_str()` | 返回 string array 的第 `i` 个字符串 |

这些 getter 大量使用 `GGML_ASSERT` 检查 id 和类型。调用错误不一定返回错误码，而可能直接终止进程。因此，读取可变来源的 GGUF 时，应先检查 `gguf_find_key()`、`gguf_get_kv_type()`、`gguf_get_arr_type()` 和数组长度。

`gguf_get_arr_str()` 内部没有独立的下标边界检查，调用者必须保证：

```text
i < gguf_get_arr_n(ctx, key_id)
```

getter 返回的字符串和数组指针指向 `gguf_context` 内部存储，在修改相应 key 或调用 `gguf_free()` 后不能继续使用。

### Tensor info 与数据偏移

每个 tensor info 在文件中包含：

```text
string   name
uint32_t n_dims
int64_t  ne[n_dims]
int32_t  ggml_type
uint64_t offset
```

`offset` 是相对 tensor data blob 起点的偏移，不是相对文件开头的绝对偏移。某个 tensor 在文件中的绝对位置为：

```c
size_t file_offset =
    gguf_get_data_offset(ctx_gguf) +
    gguf_get_tensor_offset(ctx_gguf, tensor_id);
```

相关查询接口为：

| 接口 | 作用 |
| --- | --- |
| `gguf_get_n_tensors()` | tensor 数量 |
| `gguf_find_tensor()` | 按名称查找 tensor info，未找到时返回 -1 |
| `gguf_get_tensor_name()` | tensor 名称 |
| `gguf_get_tensor_type()` | tensor 的 `ggml_type` |
| `gguf_get_tensor_offset()` | 相对 tensor data blob 的 offset |
| `gguf_get_tensor_size()` | tensor 实际数据字节数，不包含 alignment padding |
| `gguf_get_data_offset()` | tensor data blob 相对文件起点的 offset |
| `gguf_get_alignment()` | 当前文件使用的 alignment |

tensor data 区起点和每个 tensor 的起点都按 alignment 排列。alignment 默认是 `GGUF_DEFAULT_ALIGNMENT`，当前为 32；文件也可以通过 U32 类型的 `general.alignment` metadata 指定其他 2 的幂。

第 `i` 个 tensor 之后的下一个 offset 按下面的方式计算：

```text
next_offset =
    current_offset +
    GGML_PAD(current_tensor_nbytes, alignment)
```

因此要区分：

- `gguf_get_tensor_size()`：tensor 的有效数据字节数；
- tensor 占用的文件槽位：有效数据加到下一个 alignment 边界的 padding；
- `gguf_get_tensor_offset()`：相对整个 tensor data blob 的位置。

读取 tensor info 时，GGUF 会根据 shape 和 type 重建连续 stride：

```text
nb[0] = ggml_type_size(type)
nb[1] = nb[0] * ne[0] / ggml_blck_size(type)
nb[i] = nb[i - 1] * ne[i - 1], i >= 2
```

GGUF 文件不单独保存任意 stride，因此它表示的是连续 tensor 数据，不能直接持久化任意 transpose、permute 或带自定义 stride 的 view。

对于量化类型，`ne[0]` 必须是 block size 的整数倍。tensor info 中记录的 type 必须与 tensor data 区中的实际 block 格式一致；GGUF loader 不会自动量化或反量化数据。

### `gguf_context` 与 `ggml_context`

`gguf_init_from_file()` 可能同时产生两个不同对象：

| 对象 | 内容 | 释放方式 |
| --- | --- | --- |
| `gguf_context` | GGUF version、metadata、tensor info、alignment 和文件数据区位置 | `gguf_free()` |
| `ggml_context` | 根据 tensor info 创建的 `ggml_tensor` metadata，以及可选的内联 tensor data blob | `ggml_free()` |

两者不是同一个 context，生命周期也相互独立。

`gguf_free(ctx_gguf)` 只释放 GGUF metadata context，不会释放通过 `params.ctx` 返回的 `ggml_context`。反过来，`ggml_free(ctx_weights)` 也不会释放 `gguf_context`。

加载成功后，可以通过名称从返回的 `ggml_context` 中取得真正供 graph 使用的 tensor：

```c
struct ggml_tensor * weight =
    ggml_get_tensor(ctx_weights, "weight.name");
```

`gguf_get_tensor_name()` 等接口访问的是 `gguf_context` 内部的 tensor info，不返回 graph 中使用的 `ggml_tensor *`。

### 三种加载模式

加载参数为：

```c
struct gguf_init_params {
    bool no_alloc;
    struct ggml_context ** ctx;
};
```

`ctx` 和 `no_alloc` 的组合决定是否创建 tensor metadata、是否读取 tensor data：

| `params.ctx` | `no_alloc` | 行为 |
| --- | --- | --- |
| `NULL` | 任意值 | 只创建 `gguf_context`，解析 metadata 和 tensor info，不创建 `ggml_context`，也不读取 tensor data blob |
| 非 `NULL` | `true` | 创建只含 tensor metadata 的 `ggml_context`；各 tensor 的 `data` 为空，不读取 data blob |
| 非 `NULL` | `false` | 创建 `ggml_context`，分配一整块内联 data blob，读取全部 tensor data，并让各 tensor 的 `data` 指向对应 offset |

#### 直接读入主机内存

对于只使用主机地址的简单 CPU 场景，可以让 loader 一次读入全部权重：

```c
struct ggml_context * ctx_weights = NULL;

struct gguf_init_params params = {
    /*.no_alloc =*/ false,
    /*.ctx      =*/ &ctx_weights,
};

struct gguf_context * ctx_gguf =
    gguf_init_from_file("model.gguf", params);

if (ctx_gguf == NULL) {
    // 加载失败
}

struct ggml_tensor * weight =
    ggml_get_tensor(ctx_weights, "weight.name");
```

这种模式在 `ctx_weights` 中创建一个 I8 tensor 作为整个文件 data blob 的存储，然后让各权重 tensor 的 `data` 指向该 blob 内部的不同 offset。

这些权重没有独立的数据所有权。它们的数据随 `ctx_weights` 一起失效，不能单独释放。

使用完成后分别释放：

```c
gguf_free(ctx_gguf);
ggml_free(ctx_weights);
```

#### 创建 metadata 后加载到 CPU backend buffer

需要让权重由 backend buffer 管理时，可以先只创建 tensor metadata：

```c
struct ggml_context * ctx_weights = NULL;

struct gguf_init_params params = {
    /*.no_alloc =*/ true,
    /*.ctx      =*/ &ctx_weights,
};

struct gguf_context * ctx_gguf =
    gguf_init_from_file("model.gguf", params);
```

随后为这些 tensor 分配 CPU backend buffer：

```c
ggml_backend_t cpu_backend = ggml_backend_cpu_init();

ggml_backend_buffer_t weight_buffer =
    ggml_backend_alloc_ctx_tensors(ctx_weights, cpu_backend);
```

这一步只分配目标存储，不会从已经关闭的 GGUF 文件自动读取权重。应用需要重新打开文件，对每个 tensor 执行：

```text
file offset
    = gguf_get_data_offset(ctx_gguf)
    + gguf_get_tensor_offset(ctx_gguf, tensor_id)

read gguf_get_tensor_size(ctx_gguf, tensor_id) bytes
    -> ggml_backend_tensor_set(target_tensor, ...)
```

可以分块读取大 tensor，只要传给 `ggml_backend_tensor_set()` 的 tensor offset 和数据长度不越界。

这种模式把三个步骤明确分开：

```text
解析 GGUF
→ 创建 tensor metadata
→ 分配 CPU backend buffer
→ 从 GGUF 文件复制权重
```

对应生命周期也要分别管理：

```c
ggml_backend_buffer_free(weight_buffer);
ggml_free(ctx_weights);
gguf_free(ctx_gguf);
ggml_backend_free(cpu_backend);
```

### Reader 的检查与边界

`gguf_init_from_file()` 会检查：

- magic 是否为 `GGUF`；
- version 是否属于当前代码允许的范围；
- tensor 和 KV 数量是否可以表示；
- metadata key 是否重复；
- metadata type 是否有效；
- `general.alignment` 是否为非零的 2 的幂；
- tensor 名称是否重复以及是否超过 `GGML_MAX_NAME`；
- tensor 维数是否超过 `GGML_MAX_DIMS`；
- shape 和元素总数是否在允许范围内；
- `ggml_type` 是否有效；
- `ne[0]` 是否满足该类型的 block size；
- tensor offset 是否严格等于根据前面 tensor 大小和 alignment 推导出的连续 offset。

当前 reader 不接受在 data blob 中任意排列 tensor。第一个 tensor offset 必须为 0，后续 tensor 必须按照 tensor info 顺序连续排列，中间只允许规定的 alignment padding。

还需要注意两个边界：

1. 当 `params.ctx == NULL` 或 `no_alloc == true` 时，loader 不读取 tensor data blob。因此，即使文件的数据区被截断，只要 header、metadata 和 tensor info 可解析，`gguf_init_from_file()` 仍可能成功。真正读取每个 tensor 时必须检查 `fseek()` 和 `fread()` 的返回值。
2. 部分 getter 和针对保留 key 的检查使用断言或 abort。当前 GGUF API 不能视为对任意不可信文件都只返回错误码的容错解析器。

`gguf_init_from_file()` 返回后内部文件已经关闭。`gguf_context` 保存的是解析结果和 offset，不持有可继续读取的 `FILE *`。

### 创建和写出 GGUF

可以使用 `gguf_init_empty()` 创建空 context：

```c
struct gguf_context * ctx_gguf = gguf_init_empty();
```

空 context 默认使用：

```text
version   = GGUF_VERSION
alignment = GGUF_DEFAULT_ALIGNMENT
```

添加标量和 array metadata 的接口包括：

```c
gguf_set_val_str(ctx_gguf, "general.architecture", "example");
gguf_set_val_u32(ctx_gguf, "example.value", 123);

const float values[] = { 1.0f, 2.0f };
gguf_set_arr_data(
    ctx_gguf,
    "example.array",
    GGUF_TYPE_FLOAT32,
    values,
    2);
```

string array 必须使用 `gguf_set_arr_str()`；不要把 `GGUF_TYPE_STRING` 或 `GGUF_TYPE_ARRAY` 传给 `gguf_set_arr_data()`。

设置已存在的 key 时，当前实现先删除旧项，再把新项添加到 KV 列表末尾。因此，修改后原来的 key id 不再稳定；需要重新调用 `gguf_find_key()`。

`gguf_set_kv(dst, src)` 可以把另一个 context 的全部 KV 复制到目标 context。发生同名 key 时同样执行覆盖并移动到末尾。

### 添加 tensor 及其数据所有权

向 GGUF context 添加 tensor：

```c
gguf_add_tensor(ctx_gguf, tensor);
```

tensor 名称必须唯一。函数会把 `ggml_tensor` 结构浅拷贝到 `gguf_context`，并根据前一个 tensor 的大小和当前 alignment 计算 offset。

浅拷贝意味着 tensor 的 `data`、`buffer` 等底层资源不会被复制。写文件完成前，原 tensor data 或 backend buffer 必须仍然有效。

如果 tensor 使用 CPU backend buffer，writer 通过 `ggml_backend_tensor_get()` 读取数据；如果 `tensor->buffer == NULL`，writer 直接从 `tensor->data` 复制。两种情况下 tensor 都必须是连续布局，否则写入时会触发断言。

可以显式替换 writer 使用的数据地址：

```c
gguf_set_tensor_data(ctx_gguf, tensor_name, data);
```

该函数只记录指针，不复制数据。调用者必须保证：

- `data` 至少包含 `gguf_get_tensor_size()` 字节；
- 数据在写文件完成前保持有效；
- 数据的实际编码与 tensor info 中的 type 和 shape 一致。

`gguf_set_tensor_type()` 只修改 tensor info 中的类型、连续 stride 和后续 tensor offset：

```c
gguf_set_tensor_type(ctx_gguf, tensor_name, new_type);
```

它不会把原始数据转换为新类型。将 F32 tensor type 改成 Q4_0 后，调用者仍需通过 `ggml_quantize_chunk()` 生成真正的 Q4_0 block 数据，并用 `gguf_set_tensor_data()` 绑定新数据。只修改 type 会写出内容与类型不匹配的无效文件。

### 三种写出方式

#### 一次写出 metadata 和 tensor data

```c
bool ok = gguf_write_to_file(
    ctx_gguf,
    "model.gguf",
    /*only_meta =*/ false);
```

writer 会序列化 header、KV、tensor info、data 起点 padding、每个 tensor 数据及其 padding。

当前实现先在内存中的 `std::vector<int8_t>` 组装整个文件，再一次调用 `fwrite()`。因此，写大型模型时会额外占用接近完整输出文件大小的临时内存，它不是流式 writer。

#### 先写 metadata，再追加 tensor data

```c
gguf_write_to_file(
    ctx_gguf,
    "model.gguf",
    /*only_meta =*/ true);
```

`only_meta == true` 时，文件已经包含 data 区起点之前的 alignment padding，但不包含 tensor data。应用可以用 append 模式重新打开文件，按照 tensor info 顺序写入：

```text
tensor data
→ padding to alignment
→ next tensor data
→ padding to alignment
```

追加的数据布局必须与 `gguf_get_tensor_offset()` 一致。

#### 取得 metadata 二进制

也可以先取得 metadata 区大小和完整二进制内容：

```c
size_t meta_size = gguf_get_meta_size(ctx_gguf);
void * meta_data = ...; // 至少 meta_size 字节

gguf_get_meta_data(ctx_gguf, meta_data);
```

`meta_size` 已经包括 tensor data 区起点之前的 padding。这适合先预留文件头、写 tensor data，最后回填 metadata 的流程。

### `general.alignment` 的当前实现限制

加载已有文件时，reader 会读取 U32 类型的 `general.alignment`，并把它保存到 `gguf_context` 的内部 alignment 字段。

但是对于 `gguf_init_empty()` 创建的 context，当前公开 API 没有独立的 `gguf_set_alignment()`。下面的调用虽然会写入 KV：

```c
gguf_set_val_u32(
    ctx_gguf,
    GGUF_KEY_GENERAL_ALIGNMENT,
    64);
```

但当前 `gguf_set_val_u32()` 不会同步修改 `gguf_context` 内部用于计算 tensor offset 和 padding 的 alignment。这样可能导致 metadata 声明 64，而 writer 仍按默认的 32 排列数据。

因此，在当前实现修复之前，使用空 context 写文件时应保持默认 32-byte alignment，不应仅通过 `gguf_set_val_u32()` 尝试改变文件 alignment。`gguf_set_kv()` 把该 key 复制到空 context 时也存在同样问题。

### GGUF 在离线推理链路中的位置

GGUF 与其他核心模块的边界为：

```text
量化模块
    生成与 ggml_type 匹配的权重 block
        ↓
GGUF
    持久化 metadata、tensor info 和原始权重字节
        ↓
GGUF loader
    恢复 tensor metadata，并可选地读取权重
        ↓
CPU backend buffer
    保存执行时使用的权重
        ↓
computation graph + allocator + scheduler
    创建和执行前向推理
```

GGUF 负责模型数据的持久化和恢复，但不负责 graph 构建、执行期 tensor 分配、scheduler 调度或 CPU kernel 执行。

## CPU 离线推理的端到端生命周期

前面的章节分别说明了 tensor、context、graph、CPU backend、allocator、scheduler、量化和 GGUF。本节把这些模块串联为一条推荐的 CPU 离线推理路径。

这里采用以下方案：

- GGUF loader 只创建权重 tensor metadata。
- 权重存放在独立的 CPU backend buffer 中。
- graph tensor metadata 存放在单独的 graph context 中。
- scheduler 即使只有一个 CPU backend，也负责计算 tensor 的内存规划和 graph 执行。
- 同一张 graph 重复执行时不重复构图和分配。

整体关系为：

```text
GGUF 文件
    │
    ├─ gguf_context
    │      metadata、tensor info、文件 offset
    │
    └─ weights ggml_context
           权重 tensor metadata
                    │
                    ▼
           CPU weight buffer
                    │
                    ├─────────────┐
                    │             │
                    ▼             ▼
             graph context    CPU backend
                    │             │
                    └──── scheduler ────┐
                                       ▼
                              compute buffers
                                       │
                                       ▼
                                  graph output
```

### 初始化 CPU backend

首先创建 CPU backend，并在需要时配置线程数：

```c
ggml_backend_t cpu_backend = ggml_backend_cpu_init();
if (cpu_backend == NULL) {
    // 处理初始化失败。
}

ggml_backend_cpu_set_n_threads(cpu_backend, n_threads);
```

如果使用调用方创建的 threadpool，需要在 backend 销毁前解除绑定并单独释放。backend 不拥有外部 threadpool。

### 解析 GGUF 并创建权重 metadata

使用 `no_alloc == true` 读取 GGUF：

```c
struct ggml_context * ctx_weights = NULL;

struct gguf_init_params gguf_params = {
    /*.no_alloc =*/ true,
    /*.ctx      =*/ &ctx_weights,
};

struct gguf_context * ctx_gguf =
    gguf_init_from_file("model.gguf", gguf_params);

if (ctx_gguf == NULL || ctx_weights == NULL) {
    // 处理模型解析失败。
}
```

此时：

- `ctx_gguf` 保存模型 metadata、tensor info 和文件 offset。
- `ctx_weights` 保存可通过 `ggml_get_tensor()` 查找的权重 tensor metadata。
- 权重 tensor 还没有 data，也没有 backend buffer。
- GGUF 文件已经被 loader 关闭。

应用应在这个阶段检查模型架构、必要 metadata、权重名称、shape 和 type，避免把不符合预期的模型带入后续执行路径。

### 分配 CPU 权重 buffer 并加载数据

为 `ctx_weights` 中尚未分配的 tensor 创建 CPU backend buffer：

```c
ggml_backend_buffer_t weight_buffer =
    ggml_backend_alloc_ctx_tensors(ctx_weights, cpu_backend);

if (weight_buffer == NULL) {
    // 处理权重 buffer 分配失败。
}

ggml_backend_buffer_set_usage(
    weight_buffer,
    GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
```

随后重新打开 GGUF 文件。对每个 tensor：

1. 使用 `gguf_get_tensor_name()` 取得名称。
2. 使用 `ggml_get_tensor(ctx_weights, name)` 取得目标 tensor。
3. 计算文件绝对位置：

```text
gguf_get_data_offset(ctx_gguf)
    + gguf_get_tensor_offset(ctx_gguf, tensor_id)
```

4. 读取 `gguf_get_tensor_size()` 字节。
5. 使用 `ggml_backend_tensor_set()` 写入目标 tensor。

读取大 tensor 时可以使用固定大小的临时缓冲区分块复制：

```text
文件 tensor offset + chunk offset
    → temporary host buffer
    → ggml_backend_tensor_set(tensor, buffer, chunk_offset, chunk_size)
```

必须检查每次 `fseek()`、`fread()` 和目标范围。GGUF metadata-only 加载成功不代表 tensor data blob 一定完整。

全部权重复制完成后，如果后续不再查询 GGUF metadata 和文件 offset，可以立即释放：

```c
gguf_free(ctx_gguf);
ctx_gguf = NULL;
```

权重数据仍由 `weight_buffer` 持有，权重 tensor metadata 仍由 `ctx_weights` 持有。

### 创建单 CPU scheduler

将 CPU backend 交给 scheduler：

```c
ggml_backend_t backends[] = {
    cpu_backend,
};

ggml_backend_sched_t sched = ggml_backend_sched_new(
    backends,
    NULL,
    1,
    GGML_DEFAULT_GRAPH_SIZE,
    false);

if (sched == NULL) {
    // 处理 scheduler 创建失败。
}
```

在单 CPU 配置中，所有 node 都分配到 CPU，通常不会生成跨 backend copy，但 scheduler 仍然负责：

- graph tensor 的 backend 映射；
- gallocr 内存规划；
- 计算 buffer 创建和复用；
- graph 分配；
- graph 执行；
- graph reset。

scheduler 不拥有 `cpu_backend`，也不拥有 `weight_buffer`、权重 context 或用户创建的 graph context。

### 构建前向 graph

graph context 通常只分配 tensor 和 graph metadata：

```c
struct ggml_init_params graph_params = {
    /*.mem_size   =*/ graph_mem_size,
    /*.mem_buffer =*/ NULL,
    /*.no_alloc   =*/ true,
};

struct ggml_context * ctx_graph = ggml_init(graph_params);
if (ctx_graph == NULL) {
    // 处理 graph context 创建失败。
}
```

在该 context 中：

1. 创建 graph input tensor。
2. 从 `ctx_weights` 取得需要的权重 tensor。
3. 调用前向算子创建中间 tensor 和输出 tensor。
4. 标记输入和输出。
5. 从最终输出回溯构建前向 graph。

示意代码如下：

```c
struct ggml_tensor * input =
    ggml_new_tensor_2d(
        ctx_graph,
        GGML_TYPE_F32,
        input_ne0,
        input_ne1);

ggml_set_input(input);

struct ggml_tensor * weight =
    ggml_get_tensor(ctx_weights, "weight.name");

if (weight == NULL) {
    // 缺少必要权重。
}

struct ggml_tensor * output =
    build_forward(ctx_graph, input, weight);

ggml_set_output(output);

struct ggml_cgraph * graph =
    ggml_new_graph_custom(
        ctx_graph,
        GGML_DEFAULT_GRAPH_SIZE,
        false);

ggml_build_forward_expand(graph, output);
```

这里的 `build_forward()` 代表应用根据具体模型架构调用 GGML 前向算子，不是 GGML 的公共 API。

权重 tensor 可以来自另一个 context。graph 通过 tensor 指针引用这些权重，因此 `ctx_weights` 和 `weight_buffer` 必须覆盖 graph 的全部构建和执行周期。

### Reserve 与实际 graph 分配

输入 shape 存在多个档位时，可以先用能够代表最大内存需求和实际依赖关系的 measure graph 调用：

```c
bool ok = ggml_backend_sched_reserve(
    sched,
    measure_graph);
```

reserve graph 不应只拥有更大的 tensor，还应尽量保持实际 node 顺序、view、input/output 标志和生命周期关系。

对实际 graph 分配计算 tensor：

```c
if (!ggml_backend_sched_alloc_graph(sched, graph)) {
    // 处理计算 buffer 分配失败。
}
```

必须在 `alloc_graph()` 之后写 graph input，因为分配或重新 reserve 可能改变 tensor 的 `data` 地址：

```c
ggml_backend_tensor_set(
    input,
    input_data,
    0,
    ggml_nbytes(input));
```

如果没有显式调用 `ggml_backend_sched_alloc_graph()`，第一次 `ggml_backend_sched_graph_compute()` 也会自动分配。但显式分配更方便把分配失败与计算失败区分开。

### 执行并读取输出

同步执行入口为：

```c
enum ggml_status status =
    ggml_backend_sched_graph_compute(
        sched,
        graph);
```

成功后读取输出：

```c
if (status == GGML_STATUS_SUCCESS) {
    ggml_backend_tensor_get(
        output,
        output_data,
        0,
        ggml_nbytes(output));
}
```

同步入口返回时所有 scheduler backend 已经同步。若改用异步入口，则必须在读取输出、reset 或释放资源前调用：

```c
ggml_backend_sched_synchronize(sched);
```

输出必须在 scheduler reset 之前读取。reset 后，即使旧 tensor 的 `data` 字段仍保留原地址，该地址在逻辑上也已经失效。

### 重复执行同一张 graph

如果 graph 拓扑、tensor shape、view 和 backend 映射不变，可以重复使用同一张已经分配的 graph：

```c
for (int64_t i = 0; i < n_requests; ++i) {
    ggml_backend_tensor_set(
        input,
        request_data[i],
        0,
        ggml_nbytes(input));

    enum ggml_status status =
        ggml_backend_sched_graph_compute(
            sched,
            graph);

    if (status != GGML_STATUS_SUCCESS) {
        break;
    }

    ggml_backend_tensor_get(
        output,
        result_data[i],
        0,
        ggml_nbytes(output));
}
```

这种情况下不要在每次请求后 reset scheduler。输入、输出和中间 tensor 继续使用已经规划的计算 buffer。

### 更换 graph

如果需要重新构图，例如输入 shape 或 graph 拓扑发生变化，应按以下顺序操作：

```text
完成或同步上一张 graph
→ 读取仍需要的输出
→ ggml_backend_sched_reset()
→ 丢弃或 reset 旧 graph context
→ 构建新的、尚未分配的 graph
→ ggml_backend_sched_alloc_graph()
→ 写入新 graph 的 input
→ 执行
```

可以释放旧 `ctx_graph` 后创建新 context，也可以在容量足够时调用 `ggml_reset(ctx_graph)` 复用 arena。两种方式都必须先 reset scheduler，避免 scheduler 保留对旧 tensor 的分配状态。

重新 reserve 或 scheduler 扩容可能替换旧计算 buffer。旧 graph tensor 的 data 地址不能跨 reserve、reset 或扩容继续保存和使用。

### 所有权与释放顺序

主要对象的所有权如下：

| 对象 | 拥有的资源 | 不拥有的资源 |
| --- | --- | --- |
| `gguf_context` | GGUF metadata、tensor info 和内部字符串/数组 | 返回的 `ggml_context`、backend buffer |
| `ctx_weights` | 权重 tensor metadata | `weight_buffer` |
| `weight_buffer` | CPU 权重数据内存 | 权重 tensor metadata、CPU backend |
| `ctx_graph` | graph、input、output 和中间 tensor metadata | scheduler 的计算 buffer、权重 |
| scheduler | 内部 gallocr、计算 buffer、split/copy metadata | CPU backend、用户 graph、权重 buffer |
| CPU backend | CPU backend 实例及其内部执行状态 | 外部 threadpool、用户 buffer 和 context |
| 外部 threadpool | 调用者创建的工作线程 | CPU backend |

关闭整个推理实例前，应先停止提交新任务，并按以下顺序释放：

```c
// 1. 确保异步工作已经完成。
ggml_backend_sched_synchronize(sched);

// 2. 清除 scheduler 对当前 graph 的分配状态。
ggml_backend_sched_reset(sched);

// 3. 释放用户 graph metadata。
ggml_free(ctx_graph);

// 4. scheduler 释放内部 gallocr 和计算 buffer。
ggml_backend_sched_free(sched);

// 5. 释放权重数据，再释放对应的 tensor metadata。
ggml_backend_buffer_free(weight_buffer);
ggml_free(ctx_weights);

// 6. 如果 ctx_gguf 没有在权重加载后提前释放，在这里释放。
gguf_free(ctx_gguf);

// 7. 如果使用外部 threadpool，先从 backend 解除绑定，再释放 threadpool。
// ggml_backend_cpu_set_threadpool(cpu_backend, NULL);
// ggml_threadpool_free(threadpool);

// 8. 最后释放 CPU backend。
ggml_backend_free(cpu_backend);
```

若当前进程还执行过离线量化，并因此初始化了量化查找表，可以在所有量化操作结束后调用：

```c
ggml_quantize_free();
```

只加载并执行已经量化的 GGUF 权重通常不需要调用 `ggml_quantize_init()`，也不需要为此额外调用 `ggml_quantize_free()`。

最关键的生命周期边界是：

- 权重 tensor metadata 不能比引用它们的 graph 更早失效。
- 权重 buffer 不能在 graph 执行完成前释放。
- graph input 必须在 scheduler 分配完成后写入。
- output 必须在 scheduler reset 前读取。
- scheduler 必须在 CPU backend 之前释放。
- 外部 threadpool 必须在 CPU backend 不再引用它之后释放。

## 如何判断是否掌握 GGML

“能够运行示例”不等于掌握框架。示例可能隐藏了 graph 构建、buffer 所有权、内存规划和 scheduler 等关键细节。更可靠的判断标准是：能否独立解释执行链路、预测代码行为、定位故障并完成有边界的修改。

可以用下面四个层级自测：

| 层级 | 能力表现 | 建议验证任务 |
| --- | --- | --- |
| 1. 能讲清并跑通 | 能解释核心对象之间的关系，并独立完成一次 CPU 离线推理 | 不复制完整示例，完成 `GGUF -> tensor -> graph -> buffer 分配 -> CPU 执行 -> 读取输出` |
| 2. 能预测并定位 | 能根据 shape、stride、view、buffer 和 graph 判断数据如何流动，并把故障缩小到具体模块 | 定位错误结果、越界、无效 view、buffer 生命周期错误、算子不支持或 scheduler 分配错误 |
| 3. 能安全修改 | 能修改一条前向执行路径，同时处理类型、布局、内存、线程计划和回归验证 | 新增或调整一个前向算子/CPU kernel，并补充测试和错误路径检查 |
| 4. 能扩展架构 | 能实现通用接口并接入 allocator 与 scheduler，而不只是在现有代码中增加特例 | 实现一个最小实验 backend，包括 reg、device、buft、buffer 和 graph compute |

对于本文限定的“GGML 核心库 + CPU 离线部署”，达到第 3 级，并能清楚解释 scheduler，就可以认为已经掌握核心推理链路。自定义 backend 是很好的高级综合练习，但不是掌握核心库的必要条件。

### 推荐的综合验收方式

单一方式容易产生误判，建议组合完成以下任务：

1. **独立部署一次**：从模型元数据和权重加载开始，自行构建前向 graph、分配 buffer、执行并验证输出。
2. **主动制造并定位故障**：至少覆盖 tensor shape/stride、view 或所有权、allocator 容量、CPU 算子支持以及 scheduler 分配五类问题。
3. **完成一次小型修改**：选择一个前向算子或 CPU 执行细节，修改实现并用测试证明没有破坏已有路径。
4. **复盘完整生命周期**：能够说明 context、tensor 元数据、权重 buffer、计算 buffer、CPU work buffer、graph 和 scheduler 分别由谁创建、何时失效、由谁释放。

完成任务时，不只记录“最终修好了什么”，还应记录：最初假设、观测证据、排除过程、根因所在模块以及验证方法。能稳定地重复这一过程，比记住多少 API 更能说明是否真正掌握框架。

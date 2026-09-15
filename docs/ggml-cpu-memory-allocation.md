# GGML CPU 内存分配全流程：以 GPT-2 为例

本文只讨论默认 CPU backend，以 [`examples/gpt-2/main-backend.cpp`](../examples/gpt-2/main-backend.cpp) 为主线，梳理 GPT-2 从模型加载、权重分配、KV cache 分配、计算图构建到 CPU 执行期间的完整内存生命周期。

这里不展开 CUDA、Metal、Vulkan、多 backend scheduler 和跨设备 copy。

> 注意：`main-backend.cpp` 读取的是旧版 GPT-2 `ggml-model.bin`，不是 GGUF。本文先忠实说明该示例的真实路径，最后再说明它与 `GGUF + no_alloc=true` 加载方式的对应关系。

## 1. 最终会有哪些内存

只看 GGML 管理的 CPU 内存，GPT-2 推理过程中主要有六类资源：

```text
1. ctx_w->mem_buffer / ctx_kv->mem_buffer
   └── 权重与 KV tensor metadata

2. buffer_w
   └── 实际模型权重

3. buffer_kv
   └── 跨多轮 token 推理持续存在的 K/V cache

4. graph metadata buffer
   └── ggml_cgraph、输入、中间结果和 view 的 metadata

5. gallocr 的 compute buffer
   └── token input、position、计算中间结果和 logits

6. CPU backend 的 work_data
   └── CPU kernel 执行时使用的临时 scratch
```

最重要的边界是：

> `ggml_context` 主要保存 tensor 和 graph 的描述；`ggml_backend_buffer_t` 保存实际 tensor 数据；allocator 负责计算 tensor 在 buffer 中的 offset，并设置 `tensor->buffer` 和 `tensor->data`。

## 2. 先认识四个核心对象

### 2.1 `ggml_context`：metadata arena

[`ggml_context`](../src/ggml.c) 可以简化为：

```c
struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    int n_objects;
    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;
};
```

它在 `mem_buffer` 中顺序保存 `ggml_object`：

```text
[object header][tensor metadata]
[object header][tensor metadata]
[object header][graph metadata]
...
```

`no_alloc` 决定新建普通 tensor 时是否同时在 context arena 内联分配 data：

- `no_alloc == true`：只创建 tensor metadata，普通 tensor 初始 `data == NULL`；
- `no_alloc == false`：普通非 view tensor 可以在 metadata 后内联分配 data。

GPT-2 `main-backend.cpp` 对权重、KV cache 和 graph context 都使用 `no_alloc == true`，实际数据随后放进 CPU backend buffer。

### 2.2 `ggml_tensor`：描述数据及其位置

`ggml_tensor` 的相关字段为：

```c
struct ggml_tensor {
    enum ggml_type type;
    ggml_backend_buffer_t buffer;

    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];

    void * data;

    struct ggml_tensor * view_src;
    size_t view_offs;
    // op、src、name 等字段省略
};
```

其中：

- `ne[]`、`nb[]` 描述 shape 和 stride；
- `buffer` 表示数据由哪个 backend buffer 管理；
- `data` 表示该 tensor 的数据地址；
- `view_src/view_offs` 表示该 tensor 共享另一个 tensor 的存储。

一个由 `no_alloc=true` context 创建的普通 tensor 会经历：

```text
创建 metadata 后
    buffer = NULL
    data   = NULL

分配 CPU buffer 后
    buffer = CPU buffer
    data   = buffer_base + offset

写入或计算后
    buffer 和 data 不变
    data 指向的区域开始包含有效内容
```

### 2.3 CPU buffer type：分配规则

`ggml_backend_buffer_type_t`，下文简称 `buft`，描述一种内存应该如何分配。默认 CPU buft 由 `ggml_backend_cpu_buffer_type()` 返回，其主要规则为：

```text
名称              CPU
底层分配          ggml_aligned_malloc(size)
alignment         TENSOR_ALIGNMENT
get_alloc_size     默认 ggml_nbytes(tensor)
get_max_size       默认 SIZE_MAX
is_host            true
```

buft 不是实际数据，它相当于 CPU 内存分配策略和能力描述符。

### 2.4 CPU backend buffer：一次真实内存分配

`ggml_backend_buffer_t` 表示一次真实内存分配：

```c
struct ggml_backend_buffer {
    struct ggml_backend_buffer_i iface;
    ggml_backend_buffer_type_t   buft;
    void * context;
    size_t size;
    enum ggml_backend_buffer_usage usage;
};
```

默认 CPU buffer 中：

```text
buffer->context = ggml_aligned_malloc() 返回的主机地址
buffer->size    = 整块内存大小
buffer->iface   = CPU memcpy、memset 和 free 回调
```

一个 CPU buffer 可以容纳多个 tensor：

```text
CPU buffer base
├── tensor A：base + offset_A
├── padding
├── tensor B：base + offset_B
├── padding
└── tensor C：base + offset_C
```

因此 GGML 本身就是“一个数据池加多个 tensor offset”的设计。

## 3. GPT-2 模型中的内存对象

`gpt2_model` 保存三类持久对象：

```cpp
struct gpt2_model {
    struct ggml_context * ctx_w;
    struct ggml_context * ctx_kv;

    ggml_backend_t backend;

    ggml_backend_buffer_t buffer_w;
    ggml_backend_buffer_t buffer_kv;

    struct ggml_tensor * wte;
    struct ggml_tensor * wpe;
    struct ggml_tensor * lm_head;
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;
    std::vector<gpt2_layer> layers;
};
```

其关系为：

```text
ctx_w
└── 权重 tensor metadata ───────→ buffer_w 中的权重

ctx_kv
└── memory_k/v metadata ────────→ buffer_kv 中的 KV cache

graph context
└── graph tensor metadata ──────→ gallocr compute buffer
```

## 4. 阶段一：读取模型头和词表

`gpt2_model_load()` 首先打开模型文件：

```cpp
auto fin = std::ifstream(fname, std::ios::binary);
```

随后读取并检查 magic、GPT-2 hyperparameters、vocabulary 和模型权重存储类型。这些步骤主要产生普通 C++ 对象，还没有创建权重 buffer。

## 5. 阶段二：创建权重 metadata context

GPT-2 根据模型层数估算权重 tensor metadata 所需空间：

```cpp
size_t n_tensors = 2 + 6 + 12 * model.hparams.n_layer;

struct ggml_init_params params = {
    /* .mem_size   = */ ggml_tensor_overhead() * n_tensors,
    /* .mem_buffer = */ NULL,
    /* .no_alloc   = */ true,
};

model.ctx_w = ggml_init(params);
```

由于 `mem_buffer == NULL`，`ggml_init()` 会分配一个 `ggml_context` 控制结构和一块大小为 `mem_size` 的对齐 metadata arena，并设置：

```cpp
ctx_w->mem_buffer_owned = true;
ctx_w->no_alloc         = true;
```

这里的 `mem_size` 只包含 tensor metadata，不包含几百 MB 的 GPT-2 权重。

## 6. 阶段三：创建 CPU backend

当没有选择 GPU backend 时，示例进入 CPU fallback：

```cpp
if (!model.backend) {
    model.backend = ggml_backend_cpu_init();
}
```

这一步创建 CPU 执行实例，保存线程数量、可选 threadpool、可复用 `work_data`、abort callback 和 graph compute 接口。

CPU backend 不拥有模型权重；权重将在独立的 `buffer_w` 中分配。本文后续默认：

```text
n_gpu_layers = 0
model.backend = CPU backend
```

## 7. 阶段四：创建所有 GPT-2 权重 tensor

示例在 `ctx_w` 中创建顶层权重：

```cpp
model.ln_f_g = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, n_embd);
model.ln_f_b = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, n_embd);

model.wte     = ggml_new_tensor_2d(ctx_w, wtype,         n_embd, n_vocab);
model.wpe     = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_embd, n_ctx);
model.lm_head = ggml_new_tensor_2d(ctx_w, wtype,         n_embd, n_vocab);
```

每层再创建 12 个 tensor：

```text
ln_1_g / ln_1_b
ln_2_g / ln_2_b
c_attn_attn_w / c_attn_attn_b
c_attn_proj_w / c_attn_proj_b
c_mlp_fc_w / c_mlp_fc_b
c_mlp_proj_w / c_mlp_proj_b
```

`ggml_new_tensor_impl()` 只有在下面条件成立时才在 context object 后追加 data：

```cpp
if (view_src == NULL && !ctx->no_alloc) {
    obj_alloc_size = data_size;
}
```

当前 `ctx_w->no_alloc == true`，所以每个权重 tensor 创建后为：

```cpp
tensor->buffer = NULL;
tensor->data   = NULL;
```

此时 `ctx_w->mem_buffer` 仅包含所有权重的 object header 和 tensor metadata，模型数据仍在磁盘文件中。

## 8. 阶段五：一次性分配 CPU 权重池

创建所有权重 metadata 后，GPT-2 调用：

```cpp
model.buffer_w =
    ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
```

完整实现位于 [`src/ggml-alloc.c`](../src/ggml-alloc.c)。

### 8.1 统计所有权重槽位

内部先取得 CPU backend 的默认 buft，再遍历 `ctx_w` 中所有尚未分配的普通 tensor：

```cpp
size_t slot_size = GGML_PAD(
    ggml_backend_buft_get_alloc_size(cpu_buft, tensor),
    alignment
);
```

默认 CPU buft 没有自定义 `get_alloc_size`，所以：

```cpp
get_alloc_size(tensor) = ggml_nbytes(tensor);
```

假设三个 tensor 的有效数据大小为：

```text
A = 100 字节 → 对齐槽位 128 字节
B = 200 字节 → 对齐槽位 224 字节
C = 300 字节 → 对齐槽位 320 字节

total_size = 672 字节
```

GPT-2 实际权重总量要大得多，但计算方式相同。

### 8.2 分配一整块 CPU 内存

默认 CPU buft 最终执行：

```cpp
void * data = ggml_aligned_malloc(total_size);
```

然后包装成 `buffer_w`：

```text
buffer_w
├── context = CPU 内存 base
├── size    = 所有权重对齐槽位总和
├── buft    = CPU
└── iface
    ├── set_tensor     -> memcpy
    ├── get_tensor     -> memcpy
    ├── memset_tensor  -> memset
    ├── clear          -> memset
    └── free_buffer    -> ggml_aligned_free
```

### 8.3 `tallocr` 按 offset 绑定权重

`ggml_tallocr` 是单调递增分配器。对每个 tensor：

```cpp
void * addr =
    (char *) ggml_backend_buffer_get_base(buffer_w) + current_offset;

current_offset += aligned_slot_size;
ggml_backend_tensor_alloc(buffer_w, tensor, addr);
```

`ggml_backend_tensor_alloc()` 设置：

```cpp
tensor->buffer = buffer_w;
tensor->data   = addr;
```

最终成为：

```text
buffer_w base
├── ln_f_g data
├── padding
├── ln_f_b data
├── padding
├── wte data
├── padding
├── wpe data
├── 第 0 层的全部权重
├── 第 1 层的全部权重
└── ...
```

这一步之后，tensor 已有地址，但权重内容尚未从模型文件读入。

## 9. 阶段六：单独分配持久 KV cache

GPT-2 自回归推理需要保存过去 token 的 K/V。示例为它们创建独立 context：

```cpp
struct ggml_init_params params = {
    /* .mem_size   = */ 2 * ggml_tensor_overhead(),
    /* .mem_buffer = */ NULL,
    /* .no_alloc   = */ true,
};

ctx_kv = ggml_init(params);
```

然后创建两个一维 F32 tensor：

```cpp
const int64_t n_elements = n_embd * n_layer * n_ctx;

model.memory_k =
    ggml_new_tensor_1d(ctx_kv, GGML_TYPE_F32, n_elements);

model.memory_v =
    ggml_new_tensor_1d(ctx_kv, GGML_TYPE_F32, n_elements);
```

再为它们分配独立 CPU buffer：

```cpp
model.buffer_kv =
    ggml_backend_alloc_ctx_tensors(ctx_kv, model.backend);
```

结果为：

```text
buffer_kv base
├── memory_k：n_embd * n_layer * n_ctx 个 F32
├── alignment padding
└── memory_v：n_embd * n_layer * n_ctx 个 F32
```

KV cache 不放进权重池，因为两者用途不同：

| 数据 | 内容 | 可变性 | 生命周期 |
|---|---|---|---|
| `buffer_w` | 模型参数 | 推理中通常只读 | 整个模型生命周期 |
| `buffer_kv` | 已处理 token 的 K/V | 每轮推理追加写入 | 一条生成序列或会话 |

## 10. 阶段七：把模型文件权重读入 `buffer_w`

`main-backend.cpp` 的旧模型文件依次保存 tensor header 和 tensor data。loader 循环读取 tensor 的维数、名称、类型、shape 和数据，并按名称从 `model.tensors` 找到目标：

```cpp
auto tensor = model.tensors[name];
```

### 10.1 默认 CPU 路径直接读取到 `tensor->data`

因为默认 CPU buffer 是 host buffer：

```cpp
if (ggml_backend_buffer_is_host(model.buffer_w)) {
    fin.read(
        reinterpret_cast<char *>(tensor->data),
        ggml_nbytes(tensor)
    );
}
```

数据路径为：

```text
旧版 GPT-2 model.bin
        │ std::ifstream::read
        ▼
buffer_w base + 该权重的 offset
```

CPU 路径不需要额外 staging buffer，也不需要调用 `ggml_backend_tensor_set()`，因为 `tensor->data` 就是 CPU 可直接访问的主机地址。

如果统一使用通用接口，下面的调用在 CPU buffer 上最终也是 `memcpy`：

```cpp
ggml_backend_tensor_set(tensor, src, 0, ggml_nbytes(tensor));
```

### 10.2 `wte` 与 `lm_head` 的共享

GPT-2 通常让输出 head 共享 token embedding 权重。如果模型文件没有单独保存 `model/lm_head`，示例执行：

```cpp
if (name == "model/wte" && !has_lm_head) {
    model.lm_head = tensor;
}
```

这里不是创建 GGML view，而是让两个 C++ 成员指针指向同一个 `ggml_tensor`：

```text
model.wte ──────┐
                ├── 同一个 ggml_tensor ──→ buffer_w 中同一份数据
model.lm_head ──┘
```

这里还有一个容易忽略的分配细节：示例在分配 `buffer_w` 以前已经为独立的 `model/lm_head` tensor 创建了 metadata，因此 allocator 也给它划出了数据槽。随后把 `model.lm_head` 成员改为指向 `wte`，只改变了 C++ 指针，不会回收先前的 tensor metadata 和 buffer 槽位。也就是说，这是“指针别名”，不是 allocator 层面的存储合并。

## 11. 模型加载完成后的持久内存

```text
ctx_w->mem_buffer
│
├── wte metadata ─────────────────────┐
├── wpe metadata ─────────────────┐   │
├── 第 0 层权重 metadata ──────┐  │   │
└── ...                         │  │   │
                               ▼  ▼   ▼
buffer_w：CPU 权重池
├── wte 权重
├── wpe 权重
├── 第 0 层权重
├── 第 1 层权重
└── ...


ctx_kv->mem_buffer
├── memory_k metadata ─────────────┐
└── memory_v metadata ─────────┐   │
                              ▼   ▼
buffer_kv：CPU KV cache
├── memory_k data
└── memory_v data
```

## 12. 阶段八：构建 GPT-2 graph metadata

`gpt2_graph()` 准备一块只容纳 tensor 和 graph metadata 的静态数组：

```cpp
static size_t buf_size =
    ggml_tensor_overhead() * GPT2_MAX_NODES +
    ggml_graph_overhead_custom(GPT2_MAX_NODES, false);

static std::vector<uint8_t> buf(buf_size);
```

然后创建 context：

```cpp
struct ggml_init_params params = {
    /* .mem_size   = */ buf_size,
    /* .mem_buffer = */ buf.data(),
    /* .no_alloc   = */ true,
};

struct ggml_context * ctx = ggml_init(params);
```

因为 `mem_buffer` 由外部 vector 提供，所以 `ctx->mem_buffer_owned == false`；因为 `no_alloc == true`，该 context 只保存 graph 和计算 tensor metadata。

函数在完成 `ggml_build_forward_expand()` 后会先调用 `ggml_free(ctx)`，再返回 `gf`：

```cpp
ggml_build_forward_expand(gf, inpL);
ggml_free(ctx);
return gf;
```

这不会释放 `buf.data()`，因为该内存属于静态 `std::vector`，不是 context 自己拥有的。`ggml_free(ctx)` 只销毁 context 控制对象；`gf` 和各 tensor metadata 所在字节仍保留在外部 buffer 中，供后续 reserve、allocation 和 compute 使用。下一次调用 `gpt2_graph()` 会复用并重写同一块静态 metadata buffer，所以旧 graph 不能跨越下一次构图继续使用。

### 12.1 创建输入 tensor

```cpp
struct ggml_tensor * embd =
    ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);

struct ggml_tensor * position =
    ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);

ggml_set_input(embd);
ggml_set_input(position);
```

此时 `embd->data` 和 `position->data` 都是 `NULL`。输入 flag 会让 gallocr 在 graph 开始阶段为它们安排存储，并防止输入在消费前被覆盖。

### 12.2 构造算子只产生 metadata 和依赖关系

例如：

```cpp
inpL = ggml_add(
    ctx,
    ggml_get_rows(ctx, model.wte, embd),
    ggml_get_rows(ctx, model.wpe, position)
);
```

这一步只设置新 tensor 的 `op`、`src[]`、shape 和 stride，不执行 embedding lookup 或 add，也不为普通中间结果分配 data。

### 12.3 graph 引用其他 context 中的持久 tensor

GPT-2 graph 的 source 可以来自：

```text
ctx_w 中的权重 tensor
ctx_kv 中的 memory_k/memory_v
当前 graph context 中的 input 和中间 tensor
```

`ggml_cgraph` 只保存这些 tensor 指针，不取得其所有权。因此执行期间必须保证对应 metadata context 和 data buffer 都有效。

## 13. KV cache 在 graph 中如何使用

每一层会在持久 `memory_k` 和 `memory_v` 中创建当前写入位置的 view：

```cpp
struct ggml_tensor * k = ggml_view_1d(
    ctx,
    model.memory_k,
    N * n_embd,
    sizeof(float) * n_embd * (layer * n_ctx + n_past)
);
```

`v` 的处理相同。这个 view 不分配新 KV 数据，而是指向：

```text
memory_k->data + layer_offset + n_past_offset
```

随后：

```cpp
ggml_cpy(ctx, Kcur, k);
ggml_cpy(ctx, Vcur, v);
```

在 graph 执行时把当前 token 的 K/V 写入持久 `buffer_kv`。后续 token 通过 `n_past` 读取以前缓存的 K/V，并继续追加。

## 14. 阶段九：创建 gallocr 并预留计算池

主函数创建单 CPU buft 的 graph allocator：

```cpp
ggml_gallocr_t allocr = ggml_gallocr_new(
    ggml_backend_get_default_buffer_type(model.backend)
);
```

接着创建预期最坏情况的 GPT-2 graph，并显式 reserve：

```cpp
int n_tokens = std::min(model.hparams.n_ctx, params.n_batch);
int n_past   = model.hparams.n_ctx - n_tokens;

struct ggml_cgraph * graph =
    gpt2_graph(model, n_past, n_tokens);

ggml_gallocr_reserve(allocr, graph);
```

reserve 的目标是提前计算并分配峰值 compute buffer，避免每轮 token 推理反复扩容。

## 15. gallocr 如何规划 CPU 计算内存

### 15.1 跳过预分配的权重和 KV cache

权重及 KV cache 已经满足：

```cpp
tensor->data != NULL;
```

gallocr 将它们视为预分配 tensor，不会搬进 compute buffer，也不会重新安排 offset。

### 15.2 模拟临时 tensor 生命周期

尚未分配的对象主要包括 token input、position、embedding lookup 结果、每层 normalization 结果、Q/K/V 临时布局、attention score、softmax、MLP 中间结果和最终 logits。

gallocr 按 graph 顺序分析：

```text
tensor 何时产生
后面还有几个消费者
何时可以释放
释放后的区间能否被后面的 tensor 复用
```

生命周期不重叠的 tensor 可以复用同一 offset，所以所有临时 tensor 大小的总和通常远大于 compute buffer 的峰值大小。

### 15.3 保存规划并创建 CPU compute buffer

每个需要分配的 tensor 会得到类似记录：

```c
struct tensor_alloc {
    int    buffer_id;
    size_t offset;
    size_t size_max;
};
```

单 CPU buft 路径通常只有 `buffer_id == 0`。规划结束后，gallocr 按峰值调用 CPU buft：

```cpp
compute_buffer =
    ggml_backend_buft_alloc_buffer(cpu_buft, peak_size);
```

底层仍是 `ggml_aligned_malloc(peak_size)`。这块 buffer 被标记为 `GGML_BACKEND_BUFFER_USAGE_COMPUTE`。

## 16. 阶段十：为当前 graph 绑定计算地址

每次 `gpt2_eval()` 都会按当前 `n_past` 和 token 数重新构建 graph，然后调用：

```cpp
ggml_gallocr_alloc_graph(allocr, graph);
```

如果当前 graph 没有超过之前 reserve 的规划，gallocr 直接复用已有 compute buffer。

对于普通 tensor：

```cpp
void * addr =
    (char *) compute_buffer_base + planned_offset;

ggml_backend_tensor_alloc(compute_buffer, tensor, addr);
```

最终设置：

```cpp
tensor->buffer = compute_buffer;
tensor->data   = addr;
```

对于 view：

```cpp
view->buffer = view->view_src->buffer;
view->data   = (char *) view->view_src->data + view->view_offs;
```

所以 view 不占用独立的计算槽位。

## 17. 阶段十一：写入 token 和 position

graph 分配完成后才能设置输入：

```cpp
struct ggml_tensor * embd =
    ggml_graph_get_tensor(graph, "embd");

ggml_backend_tensor_set(
    embd,
    embd_inp.data(),
    0,
    N * ggml_element_size(embd)
);
```

位置输入也通过 `ggml_backend_tensor_set()` 写入。默认 CPU buffer 最终执行：

```cpp
memcpy((char *) tensor->data + offset, source, size);
```

输入进入 gallocr 的 CPU compute buffer，不会写入 `buffer_w` 或 `buffer_kv`。

## 18. 阶段十二：CPU graph 执行与 `work_data`

示例设置 CPU 线程数并执行：

```cpp
ggml_backend_cpu_set_n_threads(model.backend, n_threads);
ggml_backend_graph_compute(model.backend, graph);
```

CPU backend 内部先调用：

```cpp
struct ggml_cplan cplan =
    ggml_graph_plan(graph, n_threads, threadpool);
```

它为 CPU kernel 估算临时工作区。如果已有 `work_data` 不够大，就扩容：

```cpp
if (cpu_ctx->work_size < cplan.work_size) {
    delete[] cpu_ctx->work_data;
    cpu_ctx->work_data = new uint8_t[cplan.work_size];
    cpu_ctx->work_size = cplan.work_size;
}
```

执行期间的数据来源如下：

```text
buffer_w
    提供只读 GPT-2 权重

buffer_kv
    提供过去 token 的 K/V，并接收当前 token 的 K/V

compute buffer
    提供 embd、position 和中间结果，并接收 logits

CPU backend work_data
    提供 kernel 内部临时 scratch
```

`work_data` 不对应普通 `ggml_tensor`。CPU backend 会保留已经扩大的 `work_data`，供后续 token 推理复用；销毁 CPU backend 时才释放。

## 19. 阶段十三：读取 logits

执行完成后，示例只读取最后一个 token 的 logits：

```cpp
struct ggml_tensor * logits =
    ggml_graph_get_tensor(graph, "logits");

ggml_backend_tensor_get(
    logits,
    embd_w.data(),
    n_vocab * (N - 1) * sizeof(float),
    n_vocab * sizeof(float)
);
```

默认 CPU buffer 最终通过 `memcpy`，把 compute buffer 中的 logits 复制到应用的 `std::vector<float>`。

## 20. 为什么 CPU 设计成三个主要数据池

GPT-2 的三块主要 tensor 数据内存为：

```text
buffer_w
    长期只读的模型权重

buffer_kv
    跨 token 持续更新的 KV cache

compute buffer
    当前 graph 的输入、中间结果和输出
```

它们没有合并成一个全局 CPU 池，是因为生命周期和分配策略不同：

| 内存池 | 生命周期 | 数据特点 | 分配策略 |
|---|---|---|---|
| `buffer_w` | 模型生命周期 | 加载后通常只读 | 所有权重按顺序排布，不做生命周期复用 |
| `buffer_kv` | 当前会话/生成序列 | 每个 token 追加或更新 | 固定按层和上下文容量预留 |
| compute buffer | graph 执行资源生命周期 | 临时输入、中间值、输出 | 根据 graph 活跃区间复用 offset |

但每个池内部仍然是“一个大 buffer + 多个 offset”：

```text
权重池：buffer_w base + weight_offset
KV 池：buffer_kv base + k_or_v_offset
计算池：compute base + planned_offset
```

## 21. 默认 CPU 下是否会产生多个 `buffer_t`

`ggml_backend_alloc_ctx_tensors()` 会查询：

```cpp
size_t max_size = ggml_backend_buft_get_max_size(cpu_buft);
```

若累计大小超过单个 buffer 的最大值，它会创建下一块真实 buffer。默认 CPU buft 没有覆盖 `get_max_size`，通用默认值是 `SIZE_MAX`。

所以正常情况下：

```text
buffer_w  = 一个实际 CPU buffer
buffer_kv = 一个实际 CPU buffer
gallocr   = 一个实际 CPU compute buffer
```

这里的“三个 buffer”是因为用途和生命周期不同，不是因为单个 buffer 无法容纳多个 tensor。

## 22. 完整内存时序

```text
1. 读取 GPT-2 文件头、hparams 和 vocabulary

2. ggml_init(no_alloc=true) 创建 ctx_w
   └── 分配权重 metadata arena

3. ggml_backend_cpu_init()
   └── 创建 CPU 执行实例

4. ggml_new_tensor_*()
   └── 在 ctx_w 中创建所有权重 metadata
       buffer = NULL, data = NULL

5. ggml_backend_alloc_ctx_tensors(ctx_w, cpu_backend)
   ├── 统计所有权重的对齐槽位
   ├── 分配一整块 buffer_w
   └── 设置 weight->buffer 和 weight->data

6. 创建 memory_k 和 memory_v metadata
   └── buffer = NULL, data = NULL

7. ggml_backend_alloc_ctx_tensors(ctx_kv, cpu_backend)
   ├── 分配 buffer_kv
   └── 设置 memory_k/v->buffer 和 data

8. 从 model.bin 读取权重
   └── 直接读入 buffer_w 中各 weight->data

9. gpt2_graph()
   └── 创建 graph、input、中间 tensor 和 view metadata

10. ggml_gallocr_reserve()
    ├── 模拟 tensor 生命周期
    ├── 计算峰值
    └── 分配 CPU compute buffer

11. ggml_gallocr_alloc_graph()
    └── 根据规划绑定 input、中间结果和 logits 的 data

12. ggml_backend_tensor_set()
    └── 把 token 与 position 写入 compute buffer

13. ggml_backend_graph_compute()
    ├── 规划或复用 CPU work_data
    ├── 从 buffer_w 读取权重
    ├── 从 buffer_kv 读取并写入 K/V
    └── 在 compute buffer 中计算结果

14. ggml_backend_tensor_get()
    └── 从 compute buffer 读取最后一个 token 的 logits
```

## 23. 各类 tensor 的状态变化

### 23.1 权重 tensor

| 阶段 | `buffer` | `data` | 内容 |
|---|---|---|---|
| `ggml_new_tensor_*()` 后 | `NULL` | `NULL` | 只有 metadata |
| 分配 `buffer_w` 后 | `buffer_w` | `buffer_w base + offset` | 目标空间已分配 |
| 文件读取后 | 不变 | 不变 | 有效权重 |
| gallocr 处理 graph 后 | 不变 | 不变 | 被视为预分配，不进入 compute buffer |

### 23.2 KV tensor

| 阶段 | `buffer` | `data` | 内容 |
|---|---|---|---|
| 创建 `memory_k/v` 后 | `NULL` | `NULL` | 只有 metadata |
| 分配 `buffer_kv` 后 | `buffer_kv` | 对应槽位地址 | KV 目标空间 |
| 每次 graph 执行后 | 不变 | 不变 | 累积到当前 `n_past` 的 K/V |

### 23.3 普通 graph tensor

| 阶段 | `buffer` | `data` | 内容 |
|---|---|---|---|
| graph 构建后 | `NULL` | `NULL` | 只有 op 和依赖关系 |
| gallocr 绑定后 | compute buffer | `base + planned_offset` | 已分配但未必有有效值 |
| graph 执行后 | 不变 | 不变 | 当前执行产生的结果 |

### 23.4 View tensor

| 阶段 | 状态 |
|---|---|
| 构图时 | 保存 `view_src` 和 `view_offs`，不申请独立数据 |
| 源 tensor 已分配后 | `buffer` 继承源 buffer，`data = view_src->data + view_offs` |

## 24. 内存所有权与释放

| 资源 | 保存内容 | 释放接口 |
|---|---|---|
| `ctx_w` | 权重 tensor metadata | `ggml_free(ctx_w)` |
| `buffer_w` | 实际 CPU 权重 | `ggml_backend_buffer_free(buffer_w)` |
| `ctx_kv` | `memory_k/v` metadata | `ggml_free(ctx_kv)` |
| `buffer_kv` | 实际 KV cache | `ggml_backend_buffer_free(buffer_kv)` |
| graph context | graph 和计算 tensor metadata | `ggml_free()`；外部 metadata buffer 由调用者管理 |
| gallocr | 规划记录及 CPU compute buffer | `ggml_gallocr_free()` |
| CPU backend | 执行状态和 `work_data` | `ggml_backend_free()` |

推荐在 graph 不再执行后释放：

```cpp
ggml_gallocr_free(allocr);

ggml_backend_buffer_free(buffer_kv);
ggml_free(ctx_kv);

ggml_backend_buffer_free(buffer_w);
ggml_free(ctx_w);

ggml_backend_free(cpu_backend);
```

当前 `main-backend.cpp` 的 KV context 使用局部 `ctx` 创建，但没有把该指针正确保存并释放；示例进程退出后由操作系统回收。生产代码应把它保存在 `model.ctx_kv`，并显式调用 `ggml_free(ctx_kv)`。

## 25. 与 GGUF `no_alloc=true` 路径的对应关系

GPT-2 `main-backend.cpp` 手工创建权重 tensor metadata：

```text
读取 hparams
→ 创建 ctx_w
→ 按模型结构调用 ggml_new_tensor_*()
→ 分配 buffer_w
→ 按旧文件格式逐 tensor 读取权重
```

GGUF metadata-only 加载把前半段自动化：

```text
gguf_init_from_file(no_alloc=true, &ctx_w)
→ 根据 GGUF tensor info 自动创建权重 metadata
→ 每个 tensor 的 buffer/data 仍为 NULL

ggml_backend_alloc_ctx_tensors(ctx_w, cpu_backend)
→ 分配 CPU buffer_w
→ 设置每个 tensor 的 buffer/data

重新打开 GGUF 并按文件 offset 读取
→ 写入 tensor->data，或调用 ggml_backend_tensor_set()
```

两者在 CPU backend 分配之后的内存模型完全一致：

```text
ctx_w 保存权重 metadata
buffer_w 保存实际 CPU 权重
tensor->data = buffer_w base + tensor offset
```

区别只在于权重 metadata 和文件 offset 是由 GPT-2 loader 手工解释，还是由 GGUF loader 统一解析。

仓库中的 [`examples/gpt-2/parse_head.cpp`](../examples/gpt-2/parse_head.cpp) 可以用于观察 GPT-2 GGUF tensor info 以及 `no_alloc=true/false` 时的 context 布局，但它只负责解析和展示，不执行完整 GPT-2 推理。

## 26. CPU 内存全景图

```text
旧版 GPT-2 model.bin
        │ direct read
        ▼
buffer_w：CPU 权重池
├── embedding 权重
├── 每层 attention 权重
├── 每层 MLP 权重
└── normalization 权重
        ▲
        │ weight->data
ctx_w->mem_buffer
└── 权重 tensor metadata


buffer_kv：CPU 持久 KV cache
├── memory_k
└── memory_v
        ▲
        │ memory_k/v->data
ctx_kv->mem_buffer
└── K/V tensor metadata


graph metadata buffer
├── ggml_cgraph
├── embd / position metadata
├── 中间 tensor metadata
└── KV views
        │ gallocr 规划生命周期和 offset
        ▼
CPU compute buffer
├── embd / position data
├── 可复用中间结果区
└── logits data


CPU backend
└── work_data
    └── kernel 临时 scratch
```

## 27. 一句话复盘

GPT-2 默认 CPU 路径可以记成：

```text
创建权重壳子
→ 分配并按 offset 划分 buffer_w
→ 直接从模型文件读入权重
→ 单独分配持久 buffer_kv
→ 构建只有 metadata 的 GPT-2 graph
→ gallocr 按生命周期规划并分配 compute buffer
→ CPU backend 使用权重、KV、计算数据和 work_data 执行
```

其中：

- `ctx_w` 管权重描述；
- `buffer_w` 管长期权重；
- `ctx_kv/buffer_kv` 管跨 token 的注意力缓存；
- graph context 管计算关系；
- gallocr 的 compute buffer 管输入、中间结果和 logits；
- CPU backend 的 `work_data` 管 kernel 临时空间。

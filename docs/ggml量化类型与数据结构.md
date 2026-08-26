# `ggml-common.h` 量化类型与数据结构入门

本文面向了解基本 C 语法、但刚接触量化的读者。它以 [`src/ggml-common.h`](../src/ggml-common.h) 中的存储结构为主线：先建立直觉，再学会识别字段和宏，最后逐类阅读各种 block。

## 1. 先建立直觉：量化在做什么

模型权重通常是浮点数。如果每个权重都单独保存为 32 位浮点数，一个权重需要 4 字节。量化的基本思路是：不再逐个保存原始浮点值，而是保存较小的整数码 `q` 和一组共享元数据，使用时再近似还原。

最常见的两种还原形式是：

- 对称量化：`x_hat = d * q_signed`。`q_signed` 是以 0 为中心的有符号整数，`d` 是缩放因子（delta/scale）。
- 非对称量化（仿射量化）：`x_hat = d * q + m`。除了缩放因子 `d` 外，还有偏移 `m`，因而不必以 0 为范围中心。

例如，对称量化一组权重 `[-1.0, -0.2, 0.7, 1.1]`，假设共享 `d = 0.25`，可以取整得到 `q_signed = [-4, -1, 3, 4]`。还原值是 `[-1.0, -0.25, 0.75, 1.0]`：它们很接近原值，但不总是相等。量化因此是有损的，丢失多少与码值位数、缩放方法以及数据分布都有关。

再如，对 `[2.0, 2.5, 3.0]` 使用 `d = 0.5`、`m = 2.0` 的仿射形式，只需保存 `q = [0, 1, 2]`，就能由 `d * q + m` 还原这三个值。

GGML 不是为每个 `q` 单独保存 `d` 和 `m`，而是将一组权重组成一个 **block**，让整个 block 共享元数据。这样可以摊薄元数据的存储成本，但也意味着同一 block 内的权重必须共用某些量化参数。

K 系列还会把 256 个权重组成超块，在超块内再分小块。带最小值的 K 格式常见的还原项是 `-dmin * quantized_min`，因此其小块形式可理解为：

```text
x_hat = (d * quantized_scale) * q - dmin * quantized_min
```

这里的负号是格式规定的解码方式；不应仅根据字段名 `dmin` 就把它当成每个小块的浮点最小值。

比较格式体积时，常用 **bpw**（bits per weight，每个权重平均占用的位数）：

```text
bpw = sizeof(block) * 8 / block 表示的权重数
```

`sizeof(block)` 统计的不只是主量化码，也包含 scale、min、sign、index 等开销。例如 `block_q4_0` 表示 32 个权重，结构体共 18 字节，所以实际 `bpw = 18 * 8 / 32 = 4.5`，而不是名字中的 4。

## 2. 阅读结构体前必须知道的概念

### block 大小：`QK*` 与 `QK_K`

`QK` 的注释是“反量化后的值数量”。具体格式使用 `QK4_0`、`QK5_1`、`QK8_0` 等宏表示一个 block 可还原出多少个权重；经典 Q 系列中它们通常是 32。不要把 `QK` 误解为字节数或位数。

`QK_K` 在源码注释中定义为 super-block size，当前值为 256。K 系列常在这 256 个值内再分成多个小块，共享超块级的 `d`/`dmin`，同时为小块保存压缩后的 `scales`。不过，该头文件也复用 `QK_K` 作为 TQ 和多数 IQ 类型的 256 元素 block 长度；使用 `QK_K` 并不意味着这个类型属于 K 系列。

### 加速器打包辅助量：`QI*` 与 `QR*`

`QI*` 和 `QR*` 只在 CUDA、HIP、SYCL 等加速器分支中定义，用来描述内核如何把打包数据视为 32 位整数并成组解码。按源码注释：

- `QR*` 是反量化后值数与反量化前打包值数的比率。
- `QI*` 是反量化前以 32 位整数为单位的数量，并由类似 `QI4_0 = QK4_0 / (4 * QR4_0)` 的式子计算。其中 `4` 表示一个 32 位整数含 4 个字节。

这些宏是加速器内核的打包/解码辅助量，不是 block 中的额外字段。真正描述实际 block 内存/字节布局的是 `block_*` 结构体和它们的 `static_assert`；这些布局是否会持久化到模型权重中，要看具体类型的用途。例如 `block_q8_K` 只用于中间量化和点积。

### 常见字段名

| 字段 | 含义 |
| --- | --- |
| `d` | delta/scale。经典 block 中它直接缩放 `q`；K 超块中它还可以先解码各小块的量化 scale。 |
| `m` | min/偏移。例如 `q4_1` 使用 `d * q + m`。 |
| `dmin` | 超块中量化 min 的缩放因子。常与小块的 `quantized_min` 组成减法项，而不是单独的最小值。 |
| `qs` | 主量化数据（quants）。可能是直接的有符号码，也可能是每字节打包多个低位码或查表索引，必须结合具体格式解释。 |
| `ql` | quant 的低位部分（low bits）。 |
| `qh` | 常用于保存高位或附加编码，但准确含义依格式而定。`TQ1_0.qs` 使用三进制编码，每字节打包 5 个三值码（`3^5 = 243 < 256`）；`TQ1_0.qh` 保存剩余 16 个三值码，每字节 4 个。`IQ1_M.qh` 则混合保存 grid index 高位和 grid shift 位。 |
| `hmask` | 以位掩码形式保存 quant 的高位，例如 `q3_K` 的第 3 位。 |
| `scales` | 小块级缩放因子或 min 的量化码，也可用于 IQ 小块缩放。它本身不一定是可直接乘上权重码的浮点数。 |
| `bsums` | 每个小组的 quant 和。`q8_K` 中每 16 个 `qs` 对应一个 `int16_t` 和，便于点积内核复用。 |

同名字段在不同家族里可能有更具体的位级含义。阅读时要同时看数组长度、注释和对应的反量化函数，不要只猜字段名。

### `ggml_half` 与 `ggml_half2`

`ggml-common.h` 同时被 C、C++ 与多种 GPU 语言包含，因此 half 类型会随编译目标变化：

- C 和普通 C++ 分支将 `ggml_half` 定义为 `uint16_t`，将 `ggml_half2` 定义为 `uint32_t`。这里主要表达稳定的 16/32 位存储容器；在 CPU 代码中通过转换宏完成 half 与 float 的转换。
- Metal 使用原生 `half`/`half2`，CUDA 和 HIP 使用各自 half 头文件提供的 `half`/`half2`，SYCL 使用 `sycl::half`/`sycl::half2`，便于设备内核直接进行相应运算。

`ggml_half2` 占 32 位，可以把两个相邻 half 作为一对处理。例如结构体的联合体让 `d` 和 `m` 既能按两个 half 观察，也能通过 `dm` 作为一个 half2 观察；两种视图共享同样的字节。

### `GGML_COMMON_AGGR_U` 与 `GGML_COMMON_AGGR_S`

包含同一个头文件的语言对匿名 union/struct 的语法和警告规则不完全一样。`GGML_COMMON_AGGR_U` 和 `GGML_COMMON_AGGR_S` 会根据 C、C++、Metal、CUDA、HIP 或 SYCL 环境展开为空标识符或成员名 `data`，以便同一份结构声明能在各编译器下合法表示匿名聚合。

| 编译环境 | `GGML_COMMON_AGGR_U` | `GGML_COMMON_AGGR_S` |
| --- | --- | --- |
| C | 空 | 空 |
| C++ | `data` | `data` |
| Metal | 空 | 空 |
| CUDA / HIP / SYCL | 空 | `data` |

例如普通 C++ 允许匿名 union，但某些编译器会警告，而标准 C++ 不允许匿名 struct；该分支因此使用 `data` 给它们命名。CUDA/HIP/SYCL 分支则保留匿名 union，但给内部 struct 命名。这两个宏只是源码兼容层，不会向 block 中增加字节。

### `static_assert` 为什么紧跟每个结构体

`static_assert` 在编译期检查 `sizeof(block_*)` 是否等于格式预期的字节数。若编译器因对齐、填充或类型宽度差异改变了布局，检查就会在编译时报错，避免以错误步长读写 block。

在 C11 及以上，头文件会在需要时把 `static_assert` 映射到 `_Static_assert`；C++ 直接使用语言关键字。对缺少该能力的旧 C 环境，源码提供了兼容占位，所以那个分支不具备同等的尺寸检查保障。`static_assert` 验证的是字节布局，并不证明量化精度或解码公式正确。

## 3. 经典 Q 系列：32 个权重一个块

经典 Q 系列的六种结构都以 32 个权重为一个 block。在这个家族里，`_0` 通常表示只有 scale 的对称式布局；Q4/Q5 的 `_1` 则同时保存 scale 和 minimum，是仿射布局。`Q8_1` 是一个特例：它的第二个 FP16 不是 minimum，而是点积所需的辅助和。这套后缀规律只用于理解本节的经典 Q 结构，不能外推到 K、IQ、TQ 或其他格式。

| 结构 | block 内权重 | 结构体字节 | 实际 bpw | 编码/关键字段 |
| --- | ---: | ---: | ---: | --- |
| `block_q4_0` | 32 | 18 | 4.5 | `d` + `qs[16]` 四位码；`x_hat = d * (q - 8)` |
| `block_q4_1` | 32 | 20 | 5.0 | `d`/`m` + `qs[16]` 四位码；`x_hat = d * q + m` |
| `block_q5_0` | 32 | 22 | 5.5 | `d` + `qh[4]` 第 5 位平面 + `qs[16]` 低四位；`x_hat = d * (q - 16)` |
| `block_q5_1` | 32 | 24 | 6.0 | `d`/`m` + `qh[4]` + `qs[16]`；`x_hat = d * q + m` |
| `block_q8_0` | 32 | 34 | 8.5 | `d` + 32 个有符号 `int8_t` `qs`；`x_hat = d * q` |
| `block_q8_1` | 32 | 36 | 9.0 | `d`/`s` + 32 个有符号 `int8_t` `qs`；`s = d * sum(qs[i])` |

表中的 bpw 包含元数据。例如 `block_q5_0` 并不是每个权重恰好占 5 bit：`22 * 8 / 32 = 5.5 bpw`。

### Q4：每字节放两个四位码

`block_q4_0` 的 `d` 是一个 16 bit scale，`qs[16]` 是主量化数据。对第 `j` 个 `qs` 字节，低 nibble 是第 `j` 个权重的无符号码，高 nibble 是第 `j + 16` 个权重的码；所以同一字节中的两个 nibble 并不对应两个相邻权重。取出 `q in [0, 15]` 后先减 8 居中，再解码为 `x_hat = d * (q - 8)`。

`block_q4_1` 的联合体提供两种视图：可以分别读取 FP16 `d` 和 `m`，也可以用 `dm` 将它们作为一个 `ggml_half2` 成对处理；`dm` 不占额外空间。`qs[16]` 的 nibble 排布与 Q4_0 相同，但 `q` 不再减 8，而是按 `x_hat = d * q + m` 解码。参考量化实现将 block 的浮点 minimum 写入 `m`。

### Q5：低四位与第 5 位分开保存

`block_q5_0` 的 `d` 仍是共享 FP16 scale。`qs[16]` 与 Q4 一样，每个字节保存两个码的低四位；`qh[4]` 则合起来提供 32 bit，逻辑上第 `i` 个 bit 就是第 `i` 个码的第 5 位。解码器先组成 `q = low4 | (high1 << 4)`，得到 `q in [0, 31]`，再计算 `x_hat = d * (q - 16)`。

`block_q5_1` 在同样的 `qh[4]` + `qs[16]` 五位码之外，保存 FP16 `d` 和 `m`，并通过 `dm` 提供共享存储的 half2 视图。重组得到的 `q in [0, 31]` 不减 16，直接按 `x_hat = d * q + m` 解码。

`qh` 是独立的高位 bit-plane，不是一个元素对应一个 byte 的数组。因此不能把 `qh[i]` 当作第 `i` 个权重的高位直读；必须先按 bit 取出，再与 `qs` 里对应的低 nibble 拼接。

### Q8：主码直接是有符号整数

`block_q8_0` 保存一个 FP16 `d` 和 `int8_t qs[32]`。每个 `qs[i]` 已是一个有符号量化码，不需要拆 nibble 或重组高位，解码关系就是 `x_hat[i] = d * qs[i]`。

`block_q8_1` 也保存 FP16 `d` 和 `int8_t qs[32]`，因而单个码值同样可理解为 `d * qs[i]`。它的联合体中还有 FP16 `s`，其值由参考量化器写为 `s = d * sum(qs[i])`；`ds` 是 `d` 和 `s` 共享存储的 half2 视图，不是新字段。这个预计算的和能让带 offset/minimum 项的点积内核避免重复求和。

当前通用 type traits 为 `Q8_1` 配置了 `from_float_ref`，但没有 `to_float`；CPU/GPU 后端中可以看到它作为点积的辅助/中间格式。具体用途应以当前 checkout 的 type traits 和后端内核为准，不应仅从结构体存在就断言它一定是或不是持久化权重格式。

## 4. TQ 系列：三值量化

TQ（ternary quantization）不是用完整的 1 bit 或 2 bit 整数范围去表示权重，而是把每个有效码恢复到 `{-d, 0, +d}` 三个水平。两种 TQ 结构都以 `QK_K = 256` 个权重为一个 block，共享一个 FP16 scale `d`，但打包三值码的方法不同。

| 结构 | block 内权重 | 结构体字节 | 实际 bpw | 编码/关键字段 |
| --- | ---: | ---: | ---: | --- |
| `block_tq1_0` | 256 | 54 | 1.6875 | `qs[48]` 打包 240 个 trit，`qh[4]` 打包剩余 16 个 trit，再加 FP16 `d` |
| `block_tq2_0` | 256 | 66 | 2.0625 | `qs[64]` 每字节打包四个 2-bit code，再加 FP16 `d` |

`block_tq1_0` 的 `qs[48]` 使用三进制打包：每个 byte 逻辑上容纳 5 个 trit，因为 `3^5 = 243 < 256`。48 个 byte 因此表示 `48 * 5 = 240` 个权重。参考实现先把五个取值为 0/1/2 的 trit 合成 base-3 码，再将 `0..242` 的码映射到 8 bit 空间；解码器利用 3 的幂次逐个恢复 trit。

还剩 16 个权重，由 `qh[4]` 保存：每字节只放 4 个 trit，共 `4 * 4 = 16` 个。这里的 `qh` 是 TQ1 剩余三值码的存储区，不是 Q5 那种“第 5 位平面”。解出逻辑码 `c in {0, 1, 2}` 后，恢复式为 `x_hat = (c - 1) * d`，即 `-d`、`0`、`+d`。整个结构是 `48 + 4 + 2 = 54` 字节，所以 `54 * 8 / 256 = 1.6875 bpw`。

`block_tq2_0` 的 `qs[64]` 为每个权重分配一个 2-bit 槽位，每 byte 放四个 code，刚好覆盖 `64 * 4 = 256` 个权重。有效 code 0、1、2 经 `x_hat = (code - 1) * d` 分别恢复为 `-d`、`0`、`+d`；code 3 应视为未用编码，参考量化器不会生成它。当前参考解码循环只做 `(code - 1) * d` 而不额外验证，因而读入 code 3 会得到 `2*d`，但这不是合法的三值语义。整个结构是 `64 + 2 = 66` 字节，所以 `66 * 8 / 256 = 2.0625 bpw`。

`TQ1_0`/`TQ2_0` 名字中的 1 和 2 描述的是该格式的主码设计，不是包含 scale 和打包尾部后的物理 bpw。比较存储成本时，应使用表中由完整结构体计算的 1.6875 和 2.0625 bpw。

## 5. K 系列：256 个权重组成的超块

K 系列的 `K` 不是又一个位数，而是一组 **super-block（超块）** 格式。`QK_K = 256` 表示一个超块覆盖 256 个权重；`Q2_K`、`Q3_K` 和 `Q6_K` 再把它分成 16 个、每个 16 权重的小块，`Q4_K` 和 `Q5_K` 则分成 8 个、每个 32 权重的小块。

如果每个小块都直接保存浮点 scale/min，元数据会占用不小空间。K 系列的关键思路是：小块的 scale，以及某些格式的 min，也先量化成小整数；超块再用 FP16 `d`/`dmin` 把它们恢复成小块参数。这样能将超块级浮点元数据分摊给 256 个权重。但各格式的小块大小、主码位平面和 `scales` 打包方式不同，不能用一套位布局一式套用。

对有小块 scale/min 的格式，可以先用以下概念式理解两级缩放：

```text
sub_scale = d    * quantized_scale
sub_min   = dmin * quantized_min
x_hat     = sub_scale * q - sub_min       # 带 min 的 Q2_K/Q4_K/Q5_K
x_hat     = sub_scale * q                  # 只有 scale 的 Q3_K/Q6_K
```

这只是解码关系，不是通用的字节布局。带 min 的三种格式使用无符号 `q`；无 min 的 `Q3_K`/`Q6_K` 使用有符号 `q`，且偏移和位序必须按各自格式解读。`Q8_K` 更特殊：它没有小块 scale，而是直接用一个 FP32 `d` 缩放 256 个 `int8_t` 码。

| 结构 | 权重 | 小块划分 | 字节 | 实际 bpw | 字段构成 |
| --- | ---: | --- | ---: | ---: | --- |
| `block_q2_K` | 256 | 16 × 16 | 84 | 2.625 | `scales[16]` + `qs[64]` + FP16 `d`/`dmin`（或 `dm`） |
| `block_q3_K` | 256 | 16 × 16 | 110 | 3.4375 | `hmask[32]` + `qs[64]` + `scales[12]` + FP16 `d` |
| `block_q4_K` | 256 | 8 × 32 | 144 | 4.5 | FP16 `d`/`dmin`（或 `dm`）+ `scales[12]` + `qs[128]` |
| `block_q5_K` | 256 | 8 × 32 | 176 | 5.5 | FP16 `d`/`dmin`（或 `dm`）+ `scales[12]` + `qh[32]` + `qs[128]` |
| `block_q6_K` | 256 | 16 × 16 | 210 | 6.5625 | `ql[128]` + `qh[64]` + `scales[16]` + FP16 `d` |
| `block_q8_K` | 256 | 16 组 × 16 的求和 | 292 | 9.125 | FP32 `d` + `int8_t qs[256]` + `int16_t bsums[16]` |

### `block_q2_K`：2-bit 主码，4-bit scale/min

`scales[16]` 一个 byte 对应一个 16 权重小块：低 4 bit 是该小块的 `quantized_scale`，高 4 bit 是 `quantized_min`。`qs[64]` 为每个权重提供 2-bit 无符号码 `q in [0, 3]`；每个 byte 容纳四个 2-bit 字段/码槽。参考实现以 128 个权重为一组，同一 byte 中的四个码分别来自间隔 32 的位置，并不是四个相邻权重。

FP16 `d` 缩放低 nibble，FP16 `dmin` 缩放高 nibble，因而每个小块按 `x_hat = d * sc * q - dmin * m` 解码。结构大小是 `16 + 64 + 2 + 2 = 84` 字节。

### `block_q3_K`：2-bit 低位 + 独立高位平面

`qs[64]` 按与 Q2_K 同类的位平面顺序保存每个 3-bit 码的低 2 bit，`hmask[32]` 保存第 3 bit。`hmask` 是一个 32-byte 位平面：第一组 32 个码使用每个 byte 的 bit 0，下一组使用 bit 1，依次到 bit 7。解码时，高位为 1 就直接使用低两位，高位为 0 则减 4，得到有符号 `q in [-4, 3]`。因此这一位在实际恢复中同时承担高位/符号分组作用，不能把它当作普通二进制补码的符号位。

`scales[12]` 把 16 个 6-bit scale 压入 96 bit；解码器重组为 `0..63` 后减 32，得到 `quantized_scale in [-32, 31]`。每个 16 权重小块使用 `x_hat = d * quantized_scale * q`。结构大小为 `32 + 64 + 12 + 2 = 110` 字节。

### `block_q4_K`：4-bit 主码 + 6-bit scale/min

FP16 `d`/`dmin` 是超块参数。`scales[12]` 用刚好 96 bit 打包 8 组 `(6-bit scale, 6-bit min)`：前四组主要使用前 8 个 byte 的低 6 bit，后四组的低 4 bit 在后 4 个 byte 的两个 nibble 中，其高 2 bit 分散在前面 byte 的高位。`get_scale_min_k4` 就是负责把第 `j` 组 scale/min 重组回两个 `0..63` 整数。

`qs[128]` 每 byte 放两个 4-bit 无符号码。低 nibble 和高 nibble 分别对应同一个 64 权重分组中的前、后 32 个码，每 32 个码使用各自的小块 scale/min，按 `x_hat = d * sc * q - dmin * m` 解码。结构大小为 `4 + 12 + 128 = 144` 字节。

### `block_q5_K`：Q4_K 元数据 + 第 5 位平面

`d`/`dmin` 和 `scales[12]` 的含义、打包方式与 Q4_K 相同，也通过 `get_scale_min_k4` 重组 8 组 6-bit scale/min。`qs[128]` 保存每个码的低 4 bit，`qh[32]` 保存第 5 bit 平面；重组后得到无符号 `q in [0, 31]`。`qh` 的一个 byte 会随 64 权重分组通过不同 bit 被重用，因此不是“一权重对应一个 `qh` byte”。

每个 32 权重小块同样按 `x_hat = d * sc * q - dmin * m` 解码。结构大小是 `4 + 12 + 32 + 128 = 176` 字节。

### `block_q6_K`：4-bit 低位 + 2-bit 高位

`ql[128]` 保存 6-bit 码的低 4 bit，`qh[64]` 保存高 2 bit。打包以 128 个权重为一组：一个 `qh` byte 分别给四个间隔 32 的码提供 2 bit。高低位重组为 `0..63` 后减 32，得到有符号 `q in [-32, 31]`。

`scales[16]` 不再是压缩位流，而是每个 16 权重小块一个有符号 `int8_t` scale。FP16 `d` 是超块 scale，不使用 min，解码式为 `x_hat = d * scales[subblock] * q`。结构大小是 `128 + 64 + 16 + 2 = 210` 字节。

### `block_q8_K`：点积用的中间量化块

源码注释明确说明，`block_q8_K` **只用于中间量化和点积**。它的 `d` 是 4-byte `float`，不是 FP16；`qs[256]` 直接保存 256 个有符号 `int8_t` 码，单个值按 `x_hat = d * qs[i]` 恢复。`bsums[16]` 中的每个 `int16_t` 是对应连续 16 个 `qs` 的和，供点积内核避免重复求和；它不是额外的权重码。结构大小是 `4 + 256 + 16 * 2 = 292` 字节。

### 联合体、`scales` 与实际 bpw

`block_q2_K`、`block_q4_K` 和 `block_q5_K` 中的联合体有两种访问视图：一种将其读为两个 FP16 字段 `d`/`dmin`，另一种将同样的 4 byte 读为 `ggml_half2 dm`。它们共享内存，计算结构大小时只能计 4 byte，不能把 `dm` 再加一次。

`scales` 也不是跨格式的统一位流：Q2_K 每 byte 分成两个 4-bit 值，Q3_K 把 16 个带偏置的 6-bit scale 压成 12 byte，Q4_K/Q5_K 把 8 对 6-bit scale/min 交错打包，Q6_K 则直接使用 `int8_t scales[16]`。读写时必须使用对应格式的解码逻辑。

K 系列中没有 `Q7_K`。名字中的 2/3/4/5/6/8 描述主码设计，实际 bpw 还包含超块 scale、小块 `scales`、min、高位平面或 `bsums` 等元数据。例如 `Q4_K` 是 4.5 bpw，`Q8_K` 是 9.125 bpw，不能直接把类型名数字当成物理存储成本。

## 6. IQ 系列：查表与重要性感知量化

IQ 系列不是简单地给每个权重分配一个线性整数码。以 IQ2/IQ3 为例，量化器会把若干个权重视为一个小向量，在有限的 **grid/codebook（网格/码本）** 中选择最合适的候选向量，再结合符号与 scale 恢复近似值。因此 block 中常保存的是“选了码本中的哪一项”，而不是每个权重各自的完整整数码。

“重要性感知”描述的是**量化时的拟合策略**，不是所有 IQ 结构体中都有一个同名字段。[`ggml-quants.h`](../src/ggml-quants.h) 将 IQ 量化接口归为使用 importance matrix（也称 Activation-aware Quantization）的量化；对应量化实现会用传入的 `imatrix`/`quant_weights` 调整误差权重，从而让更重要的权重在码本搜索与 scale 拟合中获得更高优先级。是否强制需要 importance matrix 由当前量化入口决定；反量化只读 block 和程序内置表，block 里没有通用的 `importance` 字段。

九种 IQ block 也**没有统一的位布局或解码公式**。有的把 grid 索引、符号索引和小块 scale 塞进同一个 word，有的拆成 `qs`/`qh`/`signs`/`scales`，IQ4 则使用 16 值非线性码本。下表的字节数与 bpw 由结构体和其后的 `static_assert` 得到：

| 结构 | 权重 | 字节 | 实际 bpw | 完整字段 | 直观角色 |
| --- | ---: | ---: | ---: | --- | --- |
| `block_iq1_s` | 256 | 50 | 1.5625 | `d`; `qs[32]`; `uint16_t qh[8]` | `qs` 保存 grid 索引低位；`qh` 混合索引高位、grid shift 方向和小块 scale |
| `block_iq1_m` | 256 | 56 | 1.75 | `qs[32]`; `qh[16]`; `scales[8]` | 没有独立 `d`；`qs`/`qh` 组成 grid 索引与 shift，`scales` 同时打包小块 scale 和全局 FP16 scale 的 bits |
| `block_iq2_xxs` | 256 | 66 | 2.0625 | `d`; `uint16_t qs[32]` | `qs` 是打包 word，混合 grid 索引、符号索引和局部 scale |
| `block_iq2_xs` | 256 | 74 | 2.3125 | `d`; `uint16_t qs[32]`; `scales[8]` | `qs` 的 word 打包 grid/sign；`scales` 显式保存压缩的小块 scale |
| `block_iq2_s` | 256 | 82 | 2.5625 | `d`; `qs[64]`; `qh[8]`; `scales[8]` | `qs` 内部分区保存 grid 索引低位与符号，`qh` 补 grid 高位，`scales` 保存小块 scale |
| `block_iq3_xxs` | 256 | 98 | 3.0625 | `d`; `qs[96]` | 单一 byte 数组分区打包 grid 索引、符号索引和局部 scale |
| `block_iq3_s` | 256 | 110 | 3.4375 | `d`; `qs[64]`; `qh[8]`; `signs[32]`; `scales[4]` | grid 索引低/高位、符号和小块 scale 分别存放 |
| `block_iq4_nl` | 32 | 18 | 4.5 | `d`; `qs[16]` | 每个 nibble 选择 16 值非线性码本中的一项，再乘共享 `d` |
| `block_iq4_xs` | 256 | 136 | 4.25 | `d`; `uint16_t scales_h`; `scales_l[4]`; `qs[128]` | `qs` 使用 IQ4_NL nibble 码；`scales_h`/`scales_l` 打包 8 个 32 权重小块的 scale |

### IQ1：grid 索引、shift 与压缩 scale

`block_iq1_s` 每 32 个权重使用 4 个 8 维 grid 向量。`qs[32]` 为每个向量保存 8-bit 索引低位；对应的 `qh[ib]` 中，低 12 bit 依次保存 4 个索引的 3-bit 高位，bit 12–14 保存 3-bit 小块 scale，bit 15 选择在 grid 元素上加还是减 `IQ1S_DELTA`。这里的最高位是整个 32 权重小块的 grid shift 方向，不是“每个权重一个 sign bit”。FP16 `d` 再与小块 scale 组合。

`block_iq1_m` 进一步把全局 scale 也压进 `scales[8]`，所以结构体中看不到独立 `d`。反量化器把 `scales` 视为 4 个 `uint16_t`：每个 word 的低 12 bit 分成四组 3-bit 小块 scale，4 个 word 的高 nibble 则被拼回一个 FP16 全局 scale。每个 3-bit scale code 缩放两个连续的 8 维 grid，即 16 个权重；实际乘数是奇数 `2 * code + 1`。`qs` 保存索引低 8 bit，`qh` 的每个 byte 为两个 8 维 grid 向量各保存 3 个索引高位和 1 个 shift bit。它与 IQ1_S 共享 IQ1 grid 思路，但位布局和 scale 恢复步骤不同。

`iq1m_scale_t` 是 IQ1_M 解码辅助联合体：`u16` 和 `f16` 是**同一个 16-bit 存储的两种视图**。解码器先通过 `u16` 写入从 `scales` 拼出的 bits，再通过 `f16` 把这些 bits 解释为 half。它不是 block，不计入上表的九种结构；联合体也不会让 `block_iq1_m` 额外保存 2 byte。

### IQ2/IQ3：同一家族也有不同打包法

IQ2_XXS 以 32 权重为一个解码单元：其 64 bit 载荷的前 32 bit 可看成 4 个 8-bit grid 索引，后 32 bit 打包 4 个 7-bit 符号索引与 4-bit 局部 scale。`block_iq2_xxs.qs` 声明为 `uint16_t[32]`，但解码器会按 32/8-bit 视图拆这些 packed words。这里只需要先建立“索引 + 符号 + scale”的总体印象；第 7 节再按 32/8 个值逐层走读。

IQ2_XS 的每个 `uint16_t qs` 直接分为 9-bit grid 索引和 7-bit 符号索引，`scales[8]` 中每 byte 再保存两个 4-bit scale。IQ2_S 则将 10-bit grid 索引拆成 `qs` 的 8-bit 低位和 `qh` 的 2-bit 高位；`qs[64]` 的前半区域是索引低位，后半区域是直接的 8-bit 符号掩码，`scales` 另存。三者即使都属于 IQ2，也不能共用同一个位级解码式。

IQ3_XXS 把 64 byte grid 索引和 32 byte 的符号/局部 scale 区域放进同一个 `qs[96]`。IQ3_S 则把这些角色显式拆开：`qs` 存 grid 索引的低 8 bit，`qh` 补每个索引的第 9 bit，`signs` 是符号掩码，`scales` 的每个 nibble 缩放一组 32 权重。源码中 `block_iq3_s` 是 110 byte，因此实际是 `110 * 8 / 256 = 3.4375 bpw`。

### IQ4：非线性 16 值码本

`block_iq4_nl` 的一个 4-bit nibble 不是线性的 `0..15` 整数幅值，而是 `kvalues_iq4nl[16]` 的索引。当前表值从 `-127` 到 `113` 不等距分布，解码关系是 `x_hat = d * kvalues_iq4nl[code]`。`qs[16]` 每 byte 放两个 nibble，覆盖 32 个权重。

`block_iq4_xs` 复用同样的 IQ4_NL nibble code，但把 256 个权重分成 8 个 32 权重小块。每个小块的 6-bit scale 由 `scales_l` 中的低 4 bit 和 `scales_h` 中的高 2 bit 拼成，再减 32 还原成带符号小块 scale；它与全局 FP16 `d` 组合后去缩放非线性码本值。

### 后缀不是跨格式的质量等级

`XXS`/`XS`/`S` 首先是**同一数字家族内**的相对存储档，通常有 `XXS < XS < S`：例如 IQ2 依次为 2.0625、2.3125、2.5625 bpw，更大的档会用更多索引、符号或 scale bits。`M` 是另一种打包布局，`NL` 描述非线性码本；不应把它们硬塞入一条通用尺度。这些后缀不能跨 IQ1/IQ2/IQ3/IQ4 直接比较，也不能保证在任意模型、importance matrix 和硬件上的质量或速度排名。

## 7. 三个位级示例

本节将逐字节拆解 `Q4_0`、`Q4_K` 和 `IQ2_XXS`。三个例子都按“布局 → 取位 → 恢复公式 → 容量”的顺序阅读。下文的权重、子块和组索引都相对于当前 block；位编号针对整数数值，`bit 0` 是最低有效位（LSB），它本身不规定跨端序的 byte 顺序。

### 7.1 `Q4_0`：一个 byte 服务两个相隔 16 位的权重

#### 布局

一个 `block_q4_0` 保存 32 个权重，总共 18 byte：

| byte offset | 大小 | 内容 |
| --- | ---: | --- |
| `0..1` | 2 byte | FP16 缩放因子 `d` |
| `2..17` | 16 byte | `qs[0..15]`，每个 byte 放两个 4-bit code |

```text
offset 0..1             offset 2........................17
+------------------+    +---------------------------------+
| d (FP16, 2 byte) |    | qs[0] | qs[1] | ... | qs[15]   |
+------------------+    +---------------------------------+
```

#### 取位

对 `j = 0..15`，`qs[j]` 并不是保存逻辑上相邻的第 `2j` 和 `2j+1` 个权重。它的低 nibble 对应权重 `j`，高 nibble 对应权重 `j+16`：

```c
q0 = qs[j] & 0x0f;  // weight[j]
q1 = qs[j] >> 4;    // weight[j + 16]
```

例如 `qs[0] = 0xE6 = 1110 0110₂`，因此权重 0 的 code 是 `6`，权重 16 的 code 是 `14`。

#### 恢复公式

4-bit code `q` 的范围是 `0..15`，先减去固定零点 8，再乘以 `d`：

```text
x_hat = d * (q - 8)
```

纯解码示意：设 `d = 0.5`、`qs[0] = 0xE6`，则

```text
weight[0]  = 0.5 * ( 6 - 8) = -1
weight[16] = 0.5 * (14 - 8) =  3
```

`quantize_row_q4_0_ref` 有一个反直觉的细节：它找到绝对值最大的元素后，保留该元素的符号为 `max`，再计算 `d = max / -8`。所以 `d` 可能为负数；解码公式仍然是 `d * (q - 8)`。上述 `d = 0.5` 和 `0xE6` 只用于演示如何解码，不表示 reference quantizer 必然会为某组输入生成这组字节。

#### 容量

```text
(2 + 16) * 8 / 32 = 4.5 bpw
```

名字里的“4”只描述每个 code 的 4 bit；均摊 FP16 `d` 后，实际是 4.5 bpw。

### 7.2 `Q4_K`：全局因子与子块仿射参数

#### 布局

`block_q4_K` 把 256 个权重分成 8 个 32 权重子块，总共 144 byte：

| byte offset | 大小 | 内容 |
| --- | ---: | --- |
| `0..3` | 4 byte | 联合体的 `d`/`dmin` 视图：两个 FP16 全局因子（也可作 `dm` 视图） |
| `4..15` | 12 byte | `scales[12]`，打包 8 个 6-bit `sc` 和 8 个 6-bit `m` |
| `16..143` | 128 byte | `qs[128]`，保存 256 个 4-bit code |

`scales` 的位数正好对上：`8 * 6 + 8 * 6 = 96 bit = 12 byte`。但这 12 byte **不是**“8 个 scale 后跟 8 个 min”的连续数组，它们的 bits 被重组到一起。

#### 取位：`scales[12]`

令 `S` 表示 `scales`。`get_scale_min_k4(j, S, &sc, &m)` 按如下规则恢复第 `j` 个子块的两个 6-bit code：

```text
j < 4:
    sc_j = S[j]     & 0x3f
    m_j  = S[j + 4] & 0x3f

j >= 4:
    sc_j = (S[j + 4] & 0x0f) | ((S[j - 4] >> 6) << 4)
    m_j  = (S[j + 4] >> 4)   | ((S[j]     >> 6) << 4)
```

用 `j=2` 和 `j=6` 可以看清两条路径如何共享 byte（左边是高位）：

```text
S[2]  = [ sc_6 bits 5..4 ][ sc_2 bits 5..0 ]
S[6]  = [  m_6 bits 5..4 ][  m_2 bits 5..0 ]
S[10] = [  m_6 bits 3..0 ][ sc_6 bits 3..0 ]
           byte bits 7..4    byte bits 3..0
```

- 对 `j=2`，直接取 `S[2]` 和 `S[6]` 的低 6 bit，得到 `sc_2` 和 `m_2`。
- 对 `j=6`，`sc_6` 的高 2 bit 来自 `S[2]`、低 4 bit 来自 `S[10]` 的低 nibble；`m_6` 的高 2 bit 来自 `S[6]`、低 4 bit 来自 `S[10]` 的高 nibble。

#### 取位：`qs[128]`

`qs` 每 32 byte 覆盖一组 64 个逻辑权重。对组号 `g = 0..3` 和组内 byte `l = 0..31`：

```text
q_low  = qs[32*g + l] & 0x0f  -> weight[64*g + l]
q_high = qs[32*g + l] >> 4    -> weight[64*g + 32 + l]
```

因此同一 byte 的两个 nibble 分属两个不同的 32 权重子块，它们不是逻辑上相邻的权重。为了突出对应关系，把从 `scales` 恢复的逻辑 code 记作 `sc[j]` 和 `m[j]`：64 权重组 `g` 的 low nibble 属于 subblock `j=2g`，使用 `sc[2g]` 和 `m[2g]`；high nibble 属于 subblock `j=2g+1`，使用 `sc[2g+1]` 和 `m[2g+1]`。这正对应解码循环每次先取 `is+0`、再取 `is+1` 两组 metadata。

#### 恢复公式

对子块 `j`，先把 6-bit code 与全局因子组合：

```text
sub_d = d    * sc
sub_m = dmin * m
x_hat = sub_d * q - sub_m
      = (d * sc) * q - (dmin * m)
```

纯公式演示（不是某个真实 model block 的 metadata）：选用可被 FP16 精确表示的 `d=0.125` 和 `dmin=0.0625`，避免十进制示例暗含额外的 half 舍入。

```text
d = 0.125, dmin = 0.0625, sc = 10, m = 4, q = 7
sub_d = 0.125 * 10 = 1.25
sub_m = 0.0625 * 4 = 0.25
x_hat = 1.25 * 7 - 0.25 = 8.5
```

这里 `dmin * m` 是被减去，不是加上。

#### 容量

```text
144 * 8 / 256 = 4.5 bpw
```

`Q4_K` 虽然也是 4.5 bpw，但 metadata 比 `Q4_0` 分得更细：两个全局因子之下还有 8 组子块 `sc/m`。

### 7.3 `IQ2_XXS`：索引码本，而不是直接保存 2-bit 整数

#### 布局

一个 `block_iq2_xxs` 保存 256 个权重，总共 66 byte：

| byte offset | 大小 | 内容 |
| --- | ---: | --- |
| `0..1` | 2 byte | FP16 全局因子 `d` |
| `2..65` | 64 byte | `uint16_t qs[32]` 的 packed 载荷 |

256 个权重被分成 8 组，每组 32 个权重消耗 8 byte。解码器用 `memcpy` 把每组的 4 个 `uint16_t` 读成两个 `uint32_t` word。

这里的 `word0`/`word1` 是 reference C 实现执行 `memcpy` 后得到的**原生端序 `uint32_t` 视图**，不是一份跨端序的文件 byte 规范。下图中的 bit 编号以 `uint32_t` 数值的 LSB 为 0。在常见小端机器上，连续 byte 按地址从低到高恰好对应 word 从低位到高位；在大端机器上，不能直接用下图解析原始 byte，必须遵循实现与文件格式的端序处理。

```text
word0 (32 bit): 4 个 8-bit grid index
word1 (32 bit): 4 个 7-bit sign index + 1 个 4-bit local-scale code
```

`qs` 声明成 `uint16_t[32]` 是存储/对齐视图。它的语义是按 byte 和 32-bit word 打包后解码，**不是 32 个相互独立的 16-bit quant**。

#### 取位

`word0` 的 4 个 byte 分别是 `grid_index[0..3]`。每个索引进入 `iq2xxs_grid`，选出一组 8 个幅值；4 个索引刚好覆盖 32 个权重。

`word1` 的 bits 划分如下：

```text
bits  0.. 6 : sign_index[0]
bits  7..13 : sign_index[1]
bits 14..20 : sign_index[2]
bits 21..27 : sign_index[3]
bits 28..31 : local_scale_code
```

对第 `l` 组 8 个值，源码等价于：

```text
grid       = iq2xxs_grid[grid_index[l]]
sign_mask  = ksigns_iq2xs[sign_index[l]]
sign(j)    = -1, if sign_mask & kmask_iq2xs[j]
             +1, otherwise
```

`ksigns_iq2xs` 把 7-bit 索引展开为一个 8-bit 符号掩码，`kmask_iq2xs = {1, 2, 4, ..., 128}` 再检查这 8 个位置中的每一位。

#### 恢复公式

一组 32 个权重共享一个 4-bit 局部 scale code：

```text
db    = d * (0.5 + local_scale_code) * 0.25
x_hat = db * grid_value * sign
```

下面是一个与当前表值一致的小型解码示意，仍然不代表真实模型块：设 `d=0.5`、`word0=0`、`word1=0`。此时 4 个 grid index 都是 0，而 `iq2xxs_grid[0]` 的 8 个 byte 都是 8；4 个 sign index 也都是 0，`ksigns_iq2xs[0]=0`，所以全部符号为正。局部 scale code 为 0，因而：

```text
db = 0.5 * (0.5 + 0) * 0.25 = 0.0625
x_hat = 0.0625 * 8 * (+1) = 0.5
```

该示意中 32 个恢复值都是 `+0.5`。

再看一个仅在上述**小端视图**下展开的非平凡例子。仍取 `d=0.5` 和局部 scale code 0，但令第一组 8 个值的 grid index 为 1、sign index 为 1，即 `word0=0x00000001`、`word1=0x00000001`。当前 `iq2xxs_grid[1]` 常量是 `0x080808080808082b`，小端内存向量为 `[43, 8, 8, 8, 8, 8, 8, 8]`；`ksigns_iq2xs[1]=129=0b10000001`，与 `kmask_iq2xs` 配合后翻转位置 0 和 7。`db` 仍然是 `0.0625`，所以这 8 个恢复值为：

```text
[-2.6875, +0.5, +0.5, +0.5, +0.5, +0.5, +0.5, -0.5]
```

其中位置 0 为 `0.0625 * 43 * (-1) = -2.6875`，位置 7 为 `0.0625 * 8 * (-1) = -0.5`，其余位置为 `+0.5`。

#### 容量

```text
66 * 8 / 256 = 2.0625 bpw
```

`IQ2_XXS` 的额外 `0.0625 bpw` 来自均摊到 256 个权重上的 2-byte FP16 `d`。“2”并不意味着每个权重都能被当作一个独立的 2-bit 整数解码。

### 7.4 三者对比

| 格式 | 值的核心表示 | metadata 层级 | 解码直觉 |
| --- | --- | --- | --- |
| `Q4_0` | 4-bit 线性 code | 整个 32 权重块共享一个 `d` | 单一线性 scale 与固定零点 8 |
| `Q4_K` | 4-bit 非负 code | 全局 `d/dmin` + 8 组子块 `sc/m` | 分层仿射 metadata，每个子块都有 scale 和被减去的 min |
| `IQ2_XXS` | grid 索引 + 符号索引 | 全局 `d` + 每 32 值的局部 scale | 通过码本、sign 和局部 scale 共同恢复 |

## 8. 全部结构体容量速查

下表汇总 `ggml-common.h` 当前声明的全部 23 种 `block_*` 结构。“实际 bpw”统一按 `结构体字节数 * 8 / block 权重数` 计算，因此已经包含 block 内的 scale、min、sign、index 和辅助和等字段；程序全局共享的 IQ lookup table 不会随 block 重复存储，不计入 bpw。

| 家族 | 结构 | block 权重 | 字节 | 实际 bpw | metadata | 打包值 | 角色/备注 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 经典 Q | `block_q4_0` | 32 | 18 | 4.5 | FP16 `d` | `qs[16]`：32 个 nibble | 固定零点 8 的对称式 Q4 |
| 经典 Q | `block_q4_1` | 32 | 20 | 5.0 | FP16 `d`/`m`（或共享视图 `dm`） | `qs[16]`：32 个 nibble | 带 minimum 的仿射 Q4 |
| 经典 Q | `block_q5_0` | 32 | 22 | 5.5 | FP16 `d` | `qh[4]` 第 5 位 + `qs[16]` 低 4 位 | 固定零点 16 的对称式 Q5 |
| 经典 Q | `block_q5_1` | 32 | 24 | 6.0 | FP16 `d`/`m`（或共享视图 `dm`） | `qh[4]` 第 5 位 + `qs[16]` 低 4 位 | 带 minimum 的仿射 Q5 |
| 经典 Q | `block_q8_0` | 32 | 34 | 8.5 | FP16 `d` | `int8_t qs[32]` | 直接保存有符号 8-bit 码 |
| 经典 Q | `block_q8_1` | 32 | 36 | 9.0 | FP16 `d`/`s`（或共享视图 `ds`） | `int8_t qs[32]` | `s = d * sum(qs)`，用于点积辅助 |
| TQ | `block_tq1_0` | 256 | 54 | 1.6875 | FP16 `d` | `qs[48]` 每 byte 5 个 trit；`qh[4]` 存尾部 16 个 trit | 三值 `{-d, 0, +d}` 的 base-3 打包 |
| TQ | `block_tq2_0` | 256 | 66 | 2.0625 | FP16 `d` | `qs[64]`：每 byte 四个 2-bit 码槽 | 三值 `{-d, 0, +d}`；code 3 未用 |
| K | `block_q2_K` | 256 | 84 | 2.625 | FP16 `d`/`dmin`（或 `dm`）+ `scales[16]` | `qs[64]`：2-bit 无符号码 | 16 × 16 小块，scale/min 各 4 bit |
| K | `block_q3_K` | 256 | 110 | 3.4375 | FP16 `d` + `scales[12]` | `hmask[32]` 高位 + `qs[64]` 低 2 位 | 16 × 16 小块，6-bit 带偏置 scale |
| K | `block_q4_K` | 256 | 144 | 4.5 | FP16 `d`/`dmin`（或 `dm`）+ `scales[12]` | `qs[128]`：4-bit 无符号码 | 8 × 32 小块，6-bit scale/min |
| K | `block_q5_K` | 256 | 176 | 5.5 | FP16 `d`/`dmin`（或 `dm`）+ `scales[12]` | `qh[32]` 第 5 位 + `qs[128]` 低 4 位 | Q4_K metadata 加第 5 位平面 |
| K | `block_q6_K` | 256 | 210 | 6.5625 | FP16 `d` + `int8_t scales[16]` | `ql[128]` 低 4 位 + `qh[64]` 高 2 位 | 16 × 16 小块的有符号 6-bit 码 |
| K | `block_q8_K` | 256 | 292 | 9.125 | FP32 `d` + `int16_t bsums[16]` | `int8_t qs[256]` | 仅用于中间量化和点积 |
| IQ | `block_iq1_s` | 256 | 50 | 1.5625 | FP16 `d`；`qh[8]` 也打包小块 scale/shift | `qs[32]` 索引低位 + `qh[8]` 索引高位 | 每 32 权重使用四个 8 维 IQ1 grid |
| IQ | `block_iq1_m` | 256 | 56 | 1.75 | `scales[8]` 打包全局 FP16 bits 与 3-bit 小块 scale；`qh[16]` 也存 shift | `qs[32]` 索引低位 + `qh[16]` 索引高位 | 无独立 `d`；每 scale code 管 16 个权重 |
| IQ | `block_iq2_xxs` | 256 | 66 | 2.0625 | FP16 `d`；`qs` word 也打包局部 scale | `uint16_t qs[32]`：grid/sign 索引 | 每 32 权重一个 64-bit 混合载荷 |
| IQ | `block_iq2_xs` | 256 | 74 | 2.3125 | FP16 `d` + `scales[8]` | `uint16_t qs[32]`：9-bit grid + 7-bit sign 索引 | 显式 nibble 小块 scale |
| IQ | `block_iq2_s` | 256 | 82 | 2.5625 | FP16 `d` + `scales[8]` | `qs[64]` 索引低位/sign + `qh[8]` 索引高位 | 拆分 10-bit grid 索引与 sign |
| IQ | `block_iq3_xxs` | 256 | 98 | 3.0625 | FP16 `d`；`qs` 尾部也打包局部 scale | `qs[96]`：grid/sign 索引与 scale 的分区载荷 | grid、sign 与 scale 同置单一 byte 数组 |
| IQ | `block_iq3_s` | 256 | 110 | 3.4375 | FP16 `d` + `scales[4]` | `qs[64]` 索引低位 + `qh[8]` 高位 + `signs[32]` | grid、sign、scale 显式分区 |
| IQ | `block_iq4_nl` | 32 | 18 | 4.5 | FP16 `d` | `qs[16]`：32 个非线性码本 nibble 索引 | 使用 16 值 `kvalues_iq4nl` |
| IQ | `block_iq4_xs` | 256 | 136 | 4.25 | FP16 `d` + `scales_h` + `scales_l[4]` | `qs[128]`：256 个 IQ4_NL nibble 索引 | 8 × 32 小块，每块 6-bit scale |

## 9. 文件后半部分的大型表是什么

`ggml-common.h` 后半部分的大量常量是 IQ 的码本和解码辅助表。这个头文件会被 host C/C++ 和多种 GPU 后端共用，而各种语言对“常量数组”的声明语法不同。包含方在引入头文件前选择一个 `GGML_COMMON_IMPL_*` 宏，头文件再定义相应的 `GGML_TABLE_BEGIN(type, name, size)` 和 `GGML_TABLE_END()`：

| 选择宏 | 常量的主要声明形式 |
| --- | --- |
| `GGML_COMMON_IMPL_C` | host C 的 `static const`，并引入 `<stdint.h>` |
| `GGML_COMMON_IMPL_CPP` | host C++ 的 `static const`，并引入 `<cstdint>` |
| `GGML_COMMON_IMPL_METAL` | Metal 的 `static const constant` |
| `GGML_COMMON_IMPL_CUDA` / `GGML_COMMON_IMPL_HIP` / `GGML_COMMON_IMPL_MUSA` | 设备端 `static const __device__` |
| `GGML_COMMON_IMPL_SYCL` | SYCL 分支使用 `static const` |

某个分支被选中后，内部标记 `GGML_COMMON_IMPL` 会打开数据表区域。因而后面每个表只需写一次类似 `GGML_TABLE_BEGIN(uint8_t, name, size) ... GGML_TABLE_END()` 的中性形式，宏就会把它展开成当前编译目标合法的常量。这让 CPU、Metal、CUDA/HIP/MUSA 与 SYCL 内核能使用同一份数值源，避免多份表各自漂移。

几类代表表格的作用如下：

- `kmask_iq2xs` 是 `1, 2, 4, ... 128` 八个 bit mask，用来检查 8 维向量中某个元素是否要翻转符号。
- `ksigns_iq2xs` 将压缩的 7-bit 符号索引展开为 8-bit 符号组合；`ksigns64` 则把对应组合展开成八个 byte 的 `0x00`/`0xff` 形式，便于向量化内核使用。
- `iq2xxs_grid`、`iq2xs_grid`、`iq2s_grid`、`iq3xxs_grid`、`iq3s_grid` 以及 IQ1 的 grid 表是不同 IQ 布局使用的码本向量。IQ1 的 host-C 分支用 `uint64_t iq1s_grid`，其他实现分支用 `uint32_t iq1s_grid_gpu`：两者表达等价的 IQ1 grid 数值概念，但不是同一种原始数组元素表示。这项实现差异不改变 `block_iq1_s`/`block_iq1_m` 的布局；block 中的 grid index 只负责选中对应的概念向量。
- IQ4 的同类解码资产叫 `kvalues_iq4nl`，它是 16 个非线性 `int8_t` 值。要注意：在当前源码中，该表是 [`ggml-quants.c`](../src/ggml-quants.c) 内的文件局部 `static const`，而不是 `ggml-common.h` 里的 `GGML_TABLE_BEGIN/END` 表。

这些表是**实现/解码资产**，不是每个量化 block 都重复携带的 metadata。计算 `sizeof(block)` 和 bpw 时只统计结构体字段，不能把整张码本摊到每个 block 里重复加一次。阅读时理解“索引找向量、符号表展开符号”就足够建立整体模型，不需要逐项背诵上千行常量。

## 10. 推荐的源码阅读顺序

建议按下列顺序从“格式是什么”走到“运行时怎么算”：

1. 先看 [`src/ggml-common.h`](../src/ggml-common.h) 中的 `block_*` 声明、`QK*` 长度和紧跟的 `static_assert`。这一层回答“一个 block 占多少 byte、有哪些字段”，不直接回答如何取位或优化。
2. 再看 [`src/ggml-quants.h`](../src/ggml-quants.h) 中对外公开的 reference quantization 和 dequantization API，先建立“浮点行 ↔ 量化 block”的函数对应关系。
3. 在 [`src/ggml-quants.c`](../src/ggml-quants.c) 选一对 `quantize_row_*_ref` / `dequantize_row_*` 对照阅读。先读 `Q4_0` 熟悉 block 循环、nibble 和 scale，再读 `Q4_K` 的两级 metadata，最后进入 IQ 的 grid/sign/scale 查表解码。
4. 回到 [`src/ggml.c`](../src/ggml.c) 的 type traits，核对每个 `ggml_type` 的 block size、type size、reference 转换函数和 vector-dot 配对；然后再到 [`src/ggml-cpu/`](../src/ggml-cpu/)、[`src/ggml-cuda/`](../src/ggml-cuda/)、[`src/ggml-metal/`](../src/ggml-metal/) 等后端阅读 CPU/GPU 优化内核。后端是否有某个特化实现，以当前 checkout 为准。

这条路线中要始终区分三层：`ggml-common.h` 定义的是**存储布局**，`ggml-quants.c` 的 reference 函数展示的是**基准格式转换**，CPU/GPU 后端的 vector-dot 和专用内核实现的是**运行时优化计算**。三者关联同一格式，却不是同一层源码。

## 11. 常见误区

1. **“名称里的位数就是实际 bpw”。** 名义位数主要描述主码设计；实际 bpw 还包含 block 内 metadata。例如 Q4_0 是 4.5 bpw，TQ1_0 是 1.6875 bpw。
2. **“`qs` 的一个 byte 总是放相邻权重”。** Q4_0 的两个 nibble 对应间隔 16 的权重，Q2_K 同 byte 的四个 2-bit 码槽对应间隔 32 的位置。必须跟对应解码循环确认逻辑顺序。
3. **“`qh`/`ql`/`hmask` 有跨格式统一位序”。** 这些名字只提示高位、低位或位掩码的大致角色；精确分组、位移和符号语义都由具体 format 规定。
4. **“K 系列的 `dmin` 是应加上的 minimum”。** Q2_K/Q4_K/Q5_K 的解码式是 `d * sc * q - dmin * m`；`dmin` 所缩放的 min 码构成减项。
5. **“`block_q8_K` 是普通的持久化 Q8_K 权重格式”。** 头文件注释明确它只用于中间量化和点积；`bsums` 是辅助和，不是新的权重码。
6. **“IQ 就是均匀整数量化”。** IQ 码可能选择 grid/codebook、sign 组合或非线性码本值；不能把索引当作均匀的独立整数幅值直接乘 scale。
7. **“头文件共享结构声明，就表示所有后端共享同一份优化代码”。** `ggml-common.h` 共享的是格式声明和部分表数值源；CPU、CUDA、Metal 等后端仍有各自的向量化或设备内核。
8. **“lookup table 会随每个 IQ block 重复保存”。** block 里保存的是索引和局部 metadata；码本表是程序全局共享资产，不计入 `sizeof(block)` 或 bpw。
9. **“使用 `QK_K` 就属于 K 系列”。** `QK_K = 256` 被 K、TQ 和多数 IQ 结构共用为 block 长度；它是容量宏，不是家族标签。
10. **“union 中每个视图都要分别计算容量”。** `d/m` 与 `dm`、`d/s` 与 `ds`、`iq1m_scale_t.u16` 与 `.f16` 都是同一片存储的不同视图，不会重复占字节。
11. **“按 word 画的 bit 图可以无条件当成跨平台 byte 布局”。** 位移描述的是整数数值中的位；将 byte 数组通过 `memcpy` 或指针视为 `uint16_t`/`uint32_t` 时，byte 与 bit 图的对应还有端序边界。持久化数据应遵循项目的格式/加载逻辑，不能只凭本机 word 视图猜原始 byte。
12. **“更低 bpw 必然在所有模型上更准或在所有硬件上更快”。** bpw 只描述平均存储成本；质量还取决于模型和量化方法，速度还取决于当前后端是否有匹配的优化内核。

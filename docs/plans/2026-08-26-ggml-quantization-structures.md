# GGML Quantization Structures Guide Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a beginner-friendly Chinese guide to every quantization block declared in `src/ggml-common.h`, including verified storage tables and bit-level examples.

**Architecture:** Add one self-contained document under `docs/` and leave the root README and source files unchanged. Organize it from concepts to format families, then connect three representative layouts to the reference dequantization code and finish with a complete source lookup table.

**Tech Stack:** Markdown, C structure definitions in `src/ggml-common.h`, reference quantization/dequantization code in `src/ggml-quants.c`, shell-based consistency checks.

---

### Task 1: Create the guide skeleton and teach the common vocabulary

**Files:**
- Create: `docs/ggml量化类型与数据结构.md`
- Reference: `src/ggml-common.h:1-160`
- Reference: `src/ggml-quants.c:31-290`

**Step 1: Record the declared block inventory**

Run:

```bash
sed -n '1,420p' src/ggml-common.h | rg -o '^} block_[A-Za-z0-9_]+' | sort -u
```

Expected: 24 unique `block_*` declarations, from `block_iq1_m` through `block_tq2_0`.

**Step 2: Create the document outline**

Add these top-level sections in order:

```markdown
# `ggml-common.h` 量化类型与数据结构入门

## 1. 先建立直觉：量化在做什么
## 2. 阅读结构体前必须知道的概念
## 3. 经典 Q 系列：32 个权重一个块
## 4. TQ 系列：三值量化
## 5. K 系列：256 个权重组成的超块
## 6. IQ 系列：查表与重要性感知量化
## 7. 三个位级示例
## 8. 全部结构体容量速查
## 9. 文件后半部分的大型表是什么
## 10. 推荐的源码阅读顺序
## 11. 常见误区
```

**Step 3: Explain the minimum quantization model**

Cover the following points with one small numeric example:

- Quantization stores a small integer `q` plus shared metadata instead of one float per weight.
- Symmetric form: `x_hat = d * q_signed`.
- Affine form: `x_hat = d * q + m`; explain that K formats often spell the offset as `-dmin * quantized_min`.
- Quantization is lossy because `x_hat` normally differs from the original `x`.
- Block quantization shares metadata among 32 or 256 weights.
- `bpw = sizeof(block) * 8 / number_of_weights` includes scales, mins, signs, and indexes.

**Step 4: Explain shared names and portability macros**

Define `QK*`, `QK_K`, `QI*`, `QR*`, `d`, `m`, `dmin`, `qs`, `qh`, `ql`, `hmask`, `scales`, `bsums`, `ggml_half`, `ggml_half2`, `GGML_COMMON_AGGR_U`, `GGML_COMMON_AGGR_S`, and the purpose of each `static_assert`.

Make clear that `QI` and `QR` are accelerator packing helpers, while the `block_*` structures describe stored block layouts.

**Step 5: Check the skeleton and terminology**

Run:

```bash
rg -n '^## ' docs/ggml量化类型与数据结构.md
rg -n 'QK_K|bpw|static_assert|对称量化|非对称量化' docs/ggml量化类型与数据结构.md
```

Expected: all 11 numbered sections and all five foundational terms are present.

### Task 2: Document the classic Q and ternary TQ families

**Files:**
- Modify: `docs/ggml量化类型与数据结构.md`
- Reference: `src/ggml-common.h:161-241`
- Reference: `src/ggml-quants.c:31-333`
- Reference: `src/ggml-quants.c:2110-2195`

**Step 1: Add the classic Q comparison table**

Document every field and verify these values:

| Structure | Weights | Bytes | Actual bpw | Main distinction |
| --- | ---: | ---: | ---: | --- |
| `block_q4_0` | 32 | 18 | 4.5 | one FP16 scale, two 4-bit values per byte |
| `block_q4_1` | 32 | 20 | 5.0 | FP16 scale plus FP16 minimum |
| `block_q5_0` | 32 | 22 | 5.5 | low 4 bits plus a separate high-bit mask |
| `block_q5_1` | 32 | 24 | 6.0 | Q5 data plus scale and minimum |
| `block_q8_0` | 32 | 34 | 8.5 | signed 8-bit values plus FP16 scale |
| `block_q8_1` | 32 | 36 | 9.0 | also stores `s = d * sum(qs[i])` for dot products |

Explain `_0` as a scale-only symmetric-style layout and `_1` as a scale-plus-offset/auxiliary-statistic layout; do not present the suffix as a universal rule outside this family.

**Step 2: Add the TQ comparison table**

Document:

| Structure | Weights | Bytes | Actual bpw | Encoding |
| --- | ---: | ---: | ---: | --- |
| `block_tq1_0` | 256 | 54 | 1.6875 | most values packed five ternary digits per byte; remainder in `qh` |
| `block_tq2_0` | 256 | 66 | 2.0625 | four 2-bit codes per byte |

Explain that ternary values are restored from three levels and that “1-bit” in `TQ1` does not mean one physical bit per stored weight.

**Step 3: Verify family coverage**

Run:

```bash
for name in q4_0 q4_1 q5_0 q5_1 q8_0 q8_1 tq1_0 tq2_0; do rg -q "block_${name}" docs/ggml量化类型与数据结构.md || exit 1; done
```

Expected: exit status 0.

### Task 3: Document the K super-block family

**Files:**
- Modify: `docs/ggml量化类型与数据结构.md`
- Reference: `src/ggml-common.h:243-328`
- Reference: `src/ggml-quants.c:627-640`
- Reference: `src/ggml-quants.c:1208-1310`

**Step 1: Explain the two-level scale hierarchy**

Describe one 256-weight super-block split into 8 or 16 sub-blocks. Explain that per-sub-block scales/mins are themselves quantized, then reconstructed using the FP16 super-block values `d` and `dmin`.

Include the conceptual formula:

```text
sub-block scale = d * quantized_scale
sub-block min   = dmin * quantized_min
x_hat           = sub-block scale * q - sub-block min
```

Note that scale layout differs by K type; the formula is a reading aid, not a claim that every K structure has the same fields.

**Step 2: Add the K-family table**

Verify:

| Structure | Weights | Bytes | Actual bpw | Important fields |
| --- | ---: | ---: | ---: | --- |
| `block_q2_K` | 256 | 84 | 2.625 | 4-bit scale/min metadata and 2-bit codes |
| `block_q3_K` | 256 | 110 | 3.4375 | low 2 bits, high-bit mask, 6-bit scales |
| `block_q4_K` | 256 | 144 | 4.5 | 6-bit scale/min metadata and 4-bit codes |
| `block_q5_K` | 256 | 176 | 5.5 | Q4_K-like metadata plus high-bit plane |
| `block_q6_K` | 256 | 210 | 6.5625 | low 4 bits, high 2 bits, signed 8-bit scales |
| `block_q8_K` | 256 | 292 | 9.125 | float scale, signed codes, 16-element sums |

Call out `block_q8_K` as an intermediate quantization/dot-product structure, exactly as the header comment states.

**Step 3: Verify K-family coverage**

Run:

```bash
for name in q2_K q3_K q4_K q5_K q6_K q8_K; do rg -q "block_${name}" docs/ggml量化类型与数据结构.md || exit 1; done
```

Expected: exit status 0.

### Task 4: Document the IQ family and shared lookup tables

**Files:**
- Modify: `docs/ggml量化类型与数据结构.md`
- Reference: `src/ggml-common.h:330-420`
- Reference: `src/ggml-common.h:421-1853`
- Reference: `src/ggml-quants.c:2197-2310`
- Reference: `src/ggml-quants.c:2954-3140`

**Step 1: Explain how IQ differs from linear Q**

Explain that IQ formats select a vector from a discrete grid/codebook, combine it with signs and one or more scales, and use importance-aware fitting during quantization. Avoid implying that every IQ type uses identical indexes or identical tables.

Define suffixes as relative storage variants within a family: `XXS` < `XS` < `S`, while `M` is a different layout. State that suffixes are not a universal quality guarantee.

**Step 2: Add the IQ-family table**

Verify:

| Structure | Weights | Bytes | Actual bpw | Important fields |
| --- | ---: | ---: | ---: | --- |
| `block_iq1_s` | 256 | 50 | 1.5625 | low grid indexes, packed high bits, FP16 scale |
| `block_iq1_m` | 256 | 56 | 1.75 | grid indexes, shift/high bits, packed scales |
| `block_iq2_xxs` | 256 | 66 | 2.0625 | FP16 scale and packed grid/sign/scale words |
| `block_iq2_xs` | 256 | 74 | 2.3125 | grid/sign words plus explicit packed scales |
| `block_iq2_s` | 256 | 82 | 2.5625 | split index bits and per-group scales |
| `block_iq3_xxs` | 256 | 98 | 3.0625 | FP16 scale and packed 3-bit representation |
| `block_iq3_s` | 256 | 110 | 3.4375 | indexes, high bits, signs, scales |
| `block_iq4_nl` | 32 | 18 | 4.5 | non-linear 4-bit codebook plus FP16 scale |
| `block_iq4_xs` | 256 | 136 | 4.25 | IQ4 codes plus packed sub-block scales |

Explain `iq1m_scale_t` as a 16-bit bit-reinterpretation helper (`ggml_half`/`uint16_t`), not another quantized weight block.

**Step 3: Explain the implementation half of the header**

Cover `GGML_COMMON_IMPL_*`, `GGML_TABLE_BEGIN`, `GGML_TABLE_END`, `kmask_iq2xs`, `ksigns_iq2xs`, and representative `iq*grid` tables. Explain why address-space qualifiers differ among host C/C++, Metal, CUDA/HIP/MUSA, and SYCL.

**Step 4: Verify IQ coverage**

Run:

```bash
for name in iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs; do rg -q "block_${name}" docs/ggml量化类型与数据结构.md || exit 1; done
rg -q 'iq1m_scale_t' docs/ggml量化类型与数据结构.md
```

Expected: both commands exit 0.

### Task 5: Add three bit-level walkthroughs

**Files:**
- Modify: `docs/ggml量化类型与数据结构.md`
- Reference: `src/ggml-quants.c:31-60`
- Reference: `src/ggml-quants.c:255-275`
- Reference: `src/ggml-quants.c:1208-1305`
- Reference: `src/ggml-quants.c:2197-2225`
- Reference: `src/ggml-quants.c:2954-3140`

**Step 1: Walk through `Q4_0`**

Show:

- byte layout: 2-byte `d` followed by 16 `qs` bytes;
- `qs[j]` low nibble maps to weight `j`, high nibble maps to weight `j + 16`;
- decode formula `x_hat = d * (q - 8)`;
- numeric example using `d = 0.5` and `qs[0] = 0xE6`: weight 0 becomes `0.5 * (6 - 8) = -1`, weight 16 becomes `0.5 * (14 - 8) = 3`;
- capacity check `(2 + 16) * 8 / 32 = 4.5 bpw`.

Mention that the reference quantizer may store a negative `d` because it derives `d` from the signed value with maximum absolute magnitude; the decode formula remains unchanged.

**Step 2: Walk through `Q4_K`**

Show:

- layout: 4 bytes for `d/dmin`, 12 bytes for eight packed 6-bit scale/min pairs, 128 bytes for 256 4-bit codes;
- eight 32-weight sub-blocks;
- `get_scale_min_k4` recovers one 6-bit scale and one 6-bit min code;
- within each 64-weight group, a byte's low nibble addresses the first 32 values and its high nibble the next 32;
- decode formula `x_hat = (d * sc) * q - (dmin * m)`;
- capacity check `144 * 8 / 256 = 4.5 bpw`.

Use a clearly labeled illustrative numeric substitution for the formula, not a claim that those metadata values came from an actual model block.

**Step 3: Walk through `IQ2_XXS`**

Show:

- layout: 2-byte global `d` and 64 packed data bytes;
- each 32-weight group consumes two `uint32_t` words;
- the first word holds four 8-bit grid indexes;
- the second word packs four 7-bit sign indexes and a 4-bit local-scale code;
- each grid index selects eight magnitudes from `iq2xxs_grid`;
- each sign index selects eight signs from `ksigns_iq2xs`;
- `db = d * (0.5 + local_scale_code) * 0.25` and `x_hat = db * grid_value * sign`;
- capacity check `66 * 8 / 256 = 2.0625 bpw`.

Explicitly warn that `qs` is declared as `uint16_t[]` for storage/alignment but decoded through byte and 32-bit views; it is not simply an array of independent 16-bit quantized values.

**Step 4: Compare the examples**

Add a short comparison: Q4_0 uses one linear scale, Q4_K uses hierarchical affine metadata, and IQ2_XXS uses a codebook plus signs and scale. This is the key conceptual bridge for a beginner.

### Task 6: Complete the lookup table, reading route, and verification

**Files:**
- Modify: `docs/ggml量化类型与数据结构.md`
- Reference: `src/ggml-common.h:1-1853`
- Reference: `src/ggml-quants.h:1-100`
- Reference: `src/ggml-quants.c:1-3350`

**Step 1: Add the consolidated structure table**

Combine all 24 structures into one table with family, block elements, bytes, actual bpw, metadata, packed values, and role. Keep it consistent with the per-family tables rather than introducing new facts.

**Step 2: Add the source reading route**

Recommend this order:

1. declarations and size assertions in `src/ggml-common.h`;
2. public reference APIs in `src/ggml-quants.h`;
3. one `quantize_row_*_ref` and matching `dequantize_row_*` in `src/ggml-quants.c`;
4. type traits and optimized CPU/GPU kernels only after the stored layout is understood.

**Step 3: Add common mistakes**

Include at least:

- nominal bits are not actual bpw;
- a `qs` byte does not always hold adjacent logical weights;
- `qh`, `ql`, and `hmask` have format-specific bit ordering;
- `dmin` participates with a subtraction in K-family decoding;
- `Q8_K` is an intermediate/dot-product format;
- IQ layouts cannot be decoded as simple uniform integers;
- structure declarations are shared across backends, but optimized execution code lives elsewhere.

**Step 4: Verify header syntax and complete type coverage**

Run:

```bash
cc -DGGML_COMMON_DECL_C -x c -fsyntax-only src/ggml-common.h
test "$(sed -n '1,420p' src/ggml-common.h | rg -o '^} block_[A-Za-z0-9_]+' | wc -l)" -eq 24
comm -23 \
  <(sed -n '1,420p' src/ggml-common.h | rg -o 'block_[A-Za-z0-9_]+' | sort -u) \
  <(rg -o 'block_[A-Za-z0-9_]+' docs/ggml量化类型与数据结构.md | sort -u)
```

Expected: compiler exit 0, count is 24, and `comm` prints nothing.

**Step 5: Verify arithmetic and Markdown hygiene**

Manually recalculate each `bytes * 8 / weights` value from the final table, then run:

```bash
git diff --check -- docs/ggml量化类型与数据结构.md
if command -v markdownlint >/dev/null 2>&1; then markdownlint docs/ggml量化类型与数据结构.md; fi
```

Expected: no whitespace errors; Markdown lint exits 0 when installed.

**Step 6: Audit scope**

Run:

```bash
git status --short
git diff --stat -- docs/ggml量化类型与数据结构.md
```

Expected: the task adds only `docs/ggml量化类型与数据结构.md`; pre-existing modifications to `examples/tensor/nenb.cpp`, `src/ggml.c`, and `models/gpt-2-117M/ggml-model.bin` remain untouched and unstaged.

**Step 7: Commit the guide**

```bash
git add -- docs/ggml量化类型与数据结构.md
git diff --cached --check
git diff --cached --name-status
git commit -m "docs: explain ggml quantization block layouts"
```

Expected: the staged name list contains only `docs/ggml量化类型与数据结构.md`, and the commit succeeds.

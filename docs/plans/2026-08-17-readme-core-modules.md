# GGML Core Modules README Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct `readme_jht.md` against the current source and extend it into an accurate Chinese guide to GGML core modules and the CPU backend.

**Architecture:** Treat the public headers and the source files compiled into `ggml-base`, `ggml`, and `ggml-cpu` as the source of truth. Revise one module at a time using an explicit audit → user approval → patch → targeted verification checkpoint; reorder and renumber the document only after every module's content has been approved.

**Tech Stack:** C11, C++17, CMake, Markdown, Git, shell-based symbol/link checks.

---

### Task 1: Overview and module boundaries

**Files:**
- Modify: `readme_jht.md`
- Reference: `src/CMakeLists.txt`
- Reference: `src/ggml-cpu/CMakeLists.txt`

**Step 1: Prepare the audit**

Compare the current overview with the actual `ggml-base`, `ggml`, and `ggml-cpu` target sources. Prepare a table that separates data representation, graph construction, allocation, execution, persistence, quantization, and training.

**Step 2: Request approval**

Show the current problems, source evidence, and proposed overview to the user. Do not edit `readme_jht.md` until approved.

**Step 3: Apply the approved overview**

Replace the current free-form overview with the approved module map and state the document's core-library/CPU-only scope.

**Step 4: Verify**

Run `git diff --check` and check every listed source path with `test -e`.

### Task 2: Tensor, data layout, views, and operator construction

**Files:**
- Modify: `readme_jht.md`
- Reference: `include/ggml.h`
- Reference: `src/ggml.c`

**Step 1: Prepare the audit**

Trace `ggml_type`, `ggml_type_traits`, `ggml_tensor`, `ne[]`, `nb[]`, `buffer`, `data`, `op`, `op_params`, `flags`, `src[]`, `view_src`, and `view_offs`. Classify representative operators into elementwise, reduction, matrix, layout/view, neural-network, attention, and custom operations.

**Step 2: Request approval**

Present corrections and a concise proposed section, including the distinction between an operator call constructing metadata and a backend later executing it.

**Step 3: Apply only approved content**

Insert the tensor/layout/operator section without reordering unapproved modules.

**Step 4: Verify symbols**

Use `rg` against `include/ggml.h` and `src/ggml.c` for every named struct field and representative API; run `git diff --check`.

### Task 3: Context and object memory pool

**Files:**
- Modify: `readme_jht.md`
- Reference: `src/ggml.c`
- Reference: `src/gguf.cpp`

**Step 1: Prepare the audit**

Trace `ggml_init`, `ggml_new_object`, `ggml_new_tensor_impl`, `ggml_new_buffer`, `ggml_reset`, and `ggml_free`. Check the exact relationship among `ggml_context`, `ggml_object`, tensor metadata, inline tensor data, external buffers, and `no_alloc`.

**Step 2: Request approval**

Report all unsupported claims and residual junk text in the current context section, then show the replacement text and memory-layout explanation.

**Step 3: Apply the approved replacement**

Rewrite only the context section.

**Step 4: Verify**

Check field/function names against the two reference source files and run `git diff --check`.

### Task 4: Computation graph, backward graph, and graph utilities

**Files:**
- Modify: `readme_jht.md`
- Reference: `include/ggml.h`
- Reference: `src/ggml.c`

**Step 1: Prepare the audit**

Trace graph allocation, DFS visitation, node/leaf classification, forward expansion, backward expansion, gradient maps, graph views, import/export, DOT output, and reset/clear semantics.

**Step 2: Request approval**

Show corrections to the existing graph chapter and proposed additions for backward graphs and graph utilities.

**Step 3: Apply approved graph content**

Rewrite only the graph section, retaining any useful existing GPT-2 traversal example only if its path and claims remain valid.

**Step 4: Verify**

Check every graph symbol with `rg`; verify referenced local images/files exist; run `git diff --check`.

### Task 5: Backend abstraction and CPU backend

**Files:**
- Modify: `readme_jht.md`
- Reference: `include/ggml-backend.h`
- Reference: `include/ggml-cpu.h`
- Reference: `src/ggml-backend.cpp`
- Reference: `src/ggml-backend-reg.cpp`
- Reference: `src/ggml-cpu/ggml-cpu.cpp`
- Reference: `src/ggml-cpu/ggml-cpu.c`

**Step 1: Prepare the audit**

Trace registry → device → backend and buffer type → buffer → tensor relationships. For CPU, trace initialization, supported-op query, tensor copy/read/write, graph plan creation, graph compute, abort callback, and thread count/threadpool configuration.

**Step 2: Request approval**

Present inaccuracies in the current backend chapter and a CPU-only replacement. Mention other backends only as implementations excluded from scope.

**Step 3: Apply approved backend content**

Rewrite the backend section and add CPU execution details.

**Step 4: Verify**

Compile a minimal public-header CPU lifecycle snippet with the worktree build flags, then run `git diff --check`.

### Task 6: Allocation layers

**Files:**
- Modify: `readme_jht.md`
- Reference: `include/ggml-alloc.h`
- Reference: `src/ggml-alloc.c`

**Step 1: Prepare the audit**

Separate `ggml_tallocr`, `ggml_backend_alloc_ctx_tensors[_from_buft]`, `ggml_dyn_tallocr`, and `ggml_gallocr`. Verify reserve simulation, lifetime counting, in-place reuse, buffer growth, view initialization, and automatic re-reserve rules.

**Step 2: Request approval**

Show issues in the existing gallocr chapter and the proposed allocator hierarchy.

**Step 3: Apply approved allocator content**

Rewrite the gallocr chapter and add the missing lower-level allocation APIs.

**Step 4: Verify**

Check all allocator names against the public header and implementation and run `git diff --check`.

### Task 7: Backend scheduler

**Files:**
- Modify: `readme_jht.md`
- Reference: `include/ggml-backend.h`
- Reference: `src/ggml-backend.cpp`

**Step 1: Prepare the audit**

Verify scheduler construction constraints, backend assignment passes, split/copy creation, gallocr integration, reserve/allocation behavior, pipeline copies, callbacks, reset, and single-CPU degeneration.

**Step 2: Request approval**

Present corrections to the current scheduler chapter and clearly distinguish generic scheduler mechanics from the only concrete backend described, CPU.

**Step 3: Apply approved scheduler content**

Rewrite only the scheduler section.

**Step 4: Verify**

Check named fields/functions against `src/ggml-backend.cpp` and run `git diff --check`.

### Task 8: Quantization and type traits

**Files:**
- Modify: `readme_jht.md`
- Reference: `include/ggml.h`
- Reference: `src/ggml-quants.h`
- Reference: `src/ggml-quants.c`
- Reference: `src/ggml-cpu/ggml-cpu-traits.cpp`

**Step 1: Prepare the audit**

Trace block-quantized storage, type traits, row size, quantize initialization, chunk quantization, importance matrices, and the division between generic formats and CPU conversion/dot kernels.

**Step 2: Request approval**

Show the proposed conceptual section and explicitly avoid claiming all quantized types share one block layout.

**Step 3: Apply approved quantization content**

Insert the new section.

**Step 4: Verify**

Run the existing `test-quantize-fns` test and check symbols/formatting.

### Task 9: GGUF read, edit, and write

**Files:**
- Modify: `readme_jht.md`
- Reference: `include/gguf.h`
- Reference: `src/gguf.cpp`
- Reference: `docs/gguf.md`

**Step 1: Prepare the audit**

Verify serialized field widths, alignment, tensor offsets/sizes, `gguf_context` ownership, `no_alloc` behavior, tensor-data loading, metadata mutation, and the three write modes.

**Step 2: Request approval**

Present corrections to the existing GGUF section and proposed write/edit additions.

**Step 3: Apply approved GGUF content**

Rewrite only the GGUF section.

**Step 4: Verify**

Check public API symbols and compare format claims with both `include/gguf.h` and `docs/gguf.md`; run `git diff --check`.

### Task 10: Optimizer, dataset, and training flow

**Files:**
- Modify: `readme_jht.md`
- Reference: `include/ggml-opt.h`
- Reference: `src/ggml-opt.cpp`
- Reference: `tests/test-opt.cpp`

**Step 1: Prepare the audit**

Trace dataset storage/sharding, loss selection, forward/backward/optimizer graph construction, gradient accumulation, AdamW step, result aggregation, epoch, and fit helpers.

**Step 2: Request approval**

Present the proposed training-layer section and distinguish core graph autograd from the higher-level `ggml-opt` convenience module.

**Step 3: Apply approved optimizer content**

Insert the new section.

**Step 4: Verify**

Run `ctest --test-dir build -R '^test-opt$' --output-on-failure`, check symbols, and run `git diff --check`.

### Task 11: Integration, ordering, and end-to-end CPU lifecycle

**Files:**
- Modify: `readme_jht.md`

**Step 1: Prepare the integration proposal**

Show the final table of contents, proposed pure-CPU lifecycle, ownership/free order, and any content that will be moved or deduplicated.

**Step 2: Request approval**

Do not reorder, renumber, or globally rewrite the document until the user approves the final integration.

**Step 3: Apply the approved integration**

Order modules by lifecycle, normalize headings and terminology, add cross-links, remove duplicated text and the unsupported external-reference footer.

**Step 4: Validate Markdown and references**

Run checks for balanced code fences, heading hierarchy, duplicate headings, trailing whitespace errors, existing local paths, and named source symbols.

**Step 5: Run full verification**

Run `cmake --build build -j2`, then `ctest --test-dir build --output-on-failure`, followed by `git diff --check` and a final `git status --short` review.


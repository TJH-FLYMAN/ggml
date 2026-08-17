# GGML 核心模块文档修订设计

## 目标

以当前仓库源码为唯一事实依据，修正 `readme_jht.md` 中错误、过时和含混的内容，并补齐 GGML 核心库模块。文档面向希望沿调用链阅读 GGML 源码的开发者，使用中文说明，保留必要的结构体字段、关键函数和内存流转细节。

## 范围

纳入以下构建单元及其公开接口：

- `src/ggml.c`、`include/ggml.h`：tensor、context、算子、计算图、反向图、图工具。
- `src/ggml-alloc.c`、`include/ggml-alloc.h`：tensor allocator 与 graph allocator。
- `src/ggml-backend.cpp`、`src/ggml-backend-reg.cpp`、`include/ggml-backend.h`：backend 抽象、注册和 scheduler。
- `src/ggml-cpu/`、`include/ggml-cpu.h`：唯一展开说明的具体执行后端。
- `src/ggml-quants.c`、`src/ggml-quants.h`：量化类型、traits 和量化入口。
- `src/ggml-threading.cpp`：线程池、CPU 亲和性及执行线程配置。
- `src/ggml-opt.cpp`、`include/ggml-opt.h`：反向传播之上的训练、数据集和优化器封装。
- `src/gguf.cpp`、`include/gguf.h`：GGUF 读取、修改和写出。

不逐一介绍 CUDA、Metal、Vulkan、RPC 等非 CPU 后端，不展开 `examples/`，也不做逐函数 API 手册。

## 组织方式

文档按数据和执行生命周期组织：

1. 总览与构建模块边界。
2. tensor、数据类型、维度、步长和 view。
3. context 与对象内存池。
4. 算子如何产生 tensor 元数据和依赖关系。
5. 计算图、前向构图、反向图及图工具。
6. backend 抽象、buffer 与 CPU backend。
7. CPU 计算计划、线程池和 graph compute。
8. tallocr、gallocr 与 context tensor 分配。
9. scheduler；保留通用切图机制，只以 CPU 说明退化行为。
10. 量化和 type traits。
11. GGUF 读取、修改与写出。
12. optimizer、dataset 与训练调用链。
13. 纯 CPU 端到端生命周期和资源释放顺序。

算子只按类别说明，并选取能体现 shape、view、in-place 和 op 参数语义的代表 API。

## 交互式修订协议

每个模块分三步完成：

1. 只读审计，向用户列出当前文档问题、源码证据和拟写内容。
2. 等待用户明确确认或提出调整。
3. 仅把已确认模块写入 `readme_jht.md`，并运行针对该模块的符号、链接和格式检查。

不同模块不批量写入。用户未确认的模块不修改。若修订前一模块会改变全局编号，只先采用稳定标题，最终统一编号和目录。

## 准确性与验证

- 结构体字段和函数语义直接核对当前分支源码，不依赖外部博客。
- 代码示例仅使用当前公开头文件中存在的 API。
- 检查文档引用的仓库内文件、图片和符号均存在。
- 删除乱码、伪代码残片和无法由源码支持的断言。
- 全部模块完成后检查 Markdown 围栏、标题层级、重复标题、工作树 diff，并重新运行 CPU 构建与测试。


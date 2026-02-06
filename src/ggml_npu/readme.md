1. 实现设备接口（ggml_backend_device_i）
必须实现：get_name、get_description、get_memory、get_type、get_props、init_backend、get_buffer_type、supports_op、supports_buft。
可选实现：get_host_buffer_type、buffer_from_host_ptr、offload_op、事件相关接口。
示例（CANN）：设备上下文含 device、name、description；supports_op 遍历 op 类型返回是否支持 ggml-cann.cpp:2456-2524 。
2. 实现缓冲区类型与缓冲区接口
缓冲区类型（ggml_backend_buffer_type_t）实现 alloc_buffer、get_name、get_alignment 等。
缓冲区（ggml_backend_buffer_t）实现 init_tensor、set_tensor、get_tensor、memset_tensor、clear 等。
示例（ZDNN）：alloc_buffer 分页对齐分配并初始化缓冲区上下文 ggml-zdnn.cpp:327-361 。
3. 实现后端流接口（ggml_backend_i）
必须实现：get_name、free、graph_compute。
异步操作（推荐）：set_tensor_async、get_tensor_async、cpy_tensor_async、synchronize。
示例（CANN）：graph_compute 中设置设备、处理图缓存与执行 ggml-cann.cpp:2192-2237 ；异步拷贝实现见 cpy_tensor_async ggml-cann.cpp:2000-2064 。
4. 实现注册入口（ggml_backend_reg_t）
实现 ggml_backend_xxx_reg() 返回 ggml_backend_reg，包含 ggml_backend_reg_i（get_name、get_device_count、get_device、get_proc_address）。
在该函数中初始化设备列表并创建 ggml_backend_device 实例。
示例（CANN）：循环创建设备上下文并推入 vector ggml-cann.cpp:2822-2855 ；CUDA 类似 ggml-cuda.cu:4842-4885 。
5. 构建系统集成
使用 ggml_add_backend_library() 宏添加后端目标，链接 ggml-base。
若启用 GGML_BACKEND_DL，编译为 MODULE 库以便运行时加载。
参考 CANN 的 CMake 配置（检测设备、设置编译宏、链接 ACL 库）。



需要实现
1，graph_create 查询workspace\buf 存储在ggml_backend.ctx中。多个graph取max-ws。
2. buf-create
3. compute
4. ggml_tensor WHCN ->npu_tensor NCHW 步长计算
5. 插入int8->u8 



1.ggml_npu_ops.cpp实现算子相关 r853 r853_seg r853_dy

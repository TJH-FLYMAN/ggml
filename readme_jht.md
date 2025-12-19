# ggml模型解析

## ggml的核心概念
gguf : ggml模型格式
ggml_context: 一个装载各类对象 (如张量、计算图、其他数据) 的“容器”  
ggml_cgraph: 计算图的表示，可以理解为将要传给后端的“计算执行顺序”。  
ggml_backend: 执行计算图的接口，有很多种类型: CPU (默认) 、CUDA、Metal (Apple Silicon) 、Vulkan、RPC 等等  
ggml_backend_buffer_type: 表示一种缓存，可以理解为连接到每个 ggml_backend 的一个“内存分配器”。比如你要在 GPU 上执行计算，那你就需要通过一个buffer_type (通常缩写为 buft ) 去在 GPU 上分配内存  
ggml_backend_buffer: 表示一个通过 buffer_type 分配的缓存。需要注意的是，一个缓存可以存储多个张量数据。  
ggml_gallocr: 表示一个给计算图分配内存的分配器，可以给计算图中的张量进行高效的内存分配。  
ggml_backend_sched: 后端调度器，使得多种后端可以并发使用，大模型或多 GPU 推理时，实现跨硬件平台地分配计算任务 (CPU 加 GPU 混合计算、CPU和NPU混合计算)。调度器还能自动将 GPU 不支持的算子转移到 CPU 上，来确保最优的资源利用和兼容性。  

### 1.gguf 
**参考**
- docs/gguf.md
- src/gguf.*

**gguf-header**
- uint32_t magic;               GGUF的ASCII码  (G = 0x47 U 0x55  F= 0x46)  magic = 0x46554747
- uint32_t version;             GGUF版本 目前是3  0x00 0x00 0x00  0x33
- uint64_t tensor_count;        tensor数量
- uint64_t metadata_kv_count;   metadata使用键值结构kv存储
- gguf_metadata_kv_t metadata_kv[metadata_kv_count];    
    key(string)
    value : - 非数组在内存中以gguf_type(int32_t)、value_len(uint64_t)、value 存储;
            - 数组在内存中以gguf_type(int32_t),gguf_type(int32_t，后转为ggml_type)、value_len(uint64_t)、value 存储;其中第一个type为GGUF_TYPE_ARRAY，第二个type为数组内元素类型
- tensor_count 个tensor_info数据,以name_len（uint64 8字节）、name、dims_len（4字节）、dims[0]-dims[n]（ uint64 n个8字节）、datatype（uint32 4字节）、文件偏移量offset（uint64 8字节,偏移量32位对齐）
- tensor_info数据  
    name string
    n_dim uint32_t  
    ne[4] int64_t  > n_dim 的维度 为1
    type int32_t，后转为ggml_type(枚举类)
    offset uint64_t  文件偏移量，从start 开始


**gguf-model-data**
tensor_data section
offset : 读取完tensor_info数据后,当前ftell(fp) alignment对齐后为tensor数据offset;
size : 根据info中的type和shape信息，计算每个tensor数据大小(每个tensor都是alignemnet字节对齐)，累加得到size
if(!no_alloc)
    每个tensor_data持有两个head信息，ggml_object ggml_tensor
    申请size + (tensor_count+1)*[sizeof(ggml_object) + sizeof(ggml_tensor)] 字节的buf。
    tensor_data也视为ggml_object,并为ctx的头节点

**ggml_backend_reg**
参考**src/ggml-backend-reg.cpp**
- get_reg()返回全局注册表 , static ggml_backend_registry对象用于操作后端(reg load unload) 
- ggml_backend_registry定义register_backend() register_device()函数用于注册后端和设备
- ggml_backend_*_reg()函数返回 static ggml_backend_reg对象, iface实现不同。插入backends容器
- 通过backend遍历遍历后端设备返回static ggml_backend_device对象插入devices容器中(一种后端允许多个device，例如多卡gpu)

具体实现:
register_backend(ggml_backend_*_reg()) register_backend中调用register_device

使用:
ggml-backend-reg.cpp中定义了一些接口ggml_backend_reg_count \ ggml_backend_dev_count 等调用get_reg
get_reg构造函数根据编译配置自动完成后端、设备注册


**ggml_backend_cpu**

获取cpubackend
```c
//方式1 直接调用接口
ggml_backend_t backend = ggml_backend_cpu_init();

//方式2 先根据名称获取dev，根据dev初始化backend
const char* backend_name = "CPU";
ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name); // 只是检查device是否存在，返回dev
if (dev == nullptr) {
    fprintf(stderr, "%s: ERROR: backend %s not found, available:\n", __func__, backend_name.c_str());
}
ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);  // device->iface.init_backend,实际调用ggml_backend_cpu_init接口 

//方式3  自动对指定路径中的libggml-{backend_name}-*.so评分，获取best backend 。针对.so类型后端文件
ggml_backend_load_all();
ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr); 
```

**ggml_backend**
ggml_backend根据已注册的backend进行初始化，并返回ggml_backend对象

**ggml_bakcend_buffer**
ggml中一切数据（context、dataset、weight、output…）都被存放在 buffer 中。ggml使用buffer进行集成承载不同的数据，实现多种后端（CPU、GPU）设备内存的统一管理。**ggml_backend_buffer**是实现不同类型数据在多种后端上进行统一的接口对象
- struct ggml_backend_buffer_i  iface; 后端对buffer进行操作的接口 
- ggml_backend_buffer_type_t    buft; buffer所属后端类型
- void * context; 后端buffer地址
- size_t size; 后端buffer大小 
- enum ggml_backend_buffer_usage usage; buffer用途，any通用/weight权重数据/compute计算数据



**ggml_backend_cpu**
1. ggml_backend_cpu_init获取backend对象（ggml_backend）
2. ggml_backend_alloc_ctx_tensors根据1中的backend和init后的ggml_context创建ggml_backend_buffer
    - ggml_backend_get_default_buffer_type根据backend

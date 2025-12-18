# ggml模型解析

## ggml模型格式gguf 
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
register_backend(ggml_backend_*_reg())

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

# ggml

## ggml的核心概念
gguf : ggml模型格式
ggml_context: 一个装载各类对象 (如张量、计算图、其他数据) 的“容器”  
ggml_cgraph: 计算图的表示，可以理解为将要传给后端的“计算执行顺序”。  
ggml_backend: 执行计算图的接口，有很多种类型: CPU (默认) 、CUDA、Metal (Apple Silicon) 、Vulkan、RPC 等等  
ggml_backend_buffer_type: 表示一种缓存，可以理解为连接到每个 ggml_backend 的一个“内存分配器”。比如你要在 GPU 上执行计算，那你就需要通过一个buffer_type (通常缩写为 buft ) 去在 GPU 上分配内存  
ggml_backend_buffer: 表示一个通过 buffer_type 分配的缓存。需要注意的是，一个缓存可以存储多个张量数据。  
ggml_gallocr: 表示一个给计算图分配内存的分配器，可以给计算图中的张量进行高效的内存分配。  
ggml_backend_sched: 后端调度器，使得多种后端可以并发使用，大模型或多 GPU 推理时，实现跨硬件平台地分配计算任务 (CPU 加 GPU 混合计算、CPU和NPU混合计算)。调度器还能自动将 GPU 不支持的算子转移到 CPU 上，来确保最优的资源利用和兼容性。  

## 1. GGUF
gguf模型解析，保存到gguf_context
### GGUF文件结构
```
[Header] -> [Metadata] -> [Tensor Info] -> [Tensor Data]
```
**gguf-header**  
- magic;           GGUF的ASCII码 0x46554747  
- version;         GGUF版本 目前是3  
- n_tensor;        tensor数量  
- n_kv;            metadata数量(kv对数量)    

**metadata**  
- gguf_metadata_kv_t metadata_kv[metadata_kv_count];      
key(string)  
value   
    a. NO_array元素在内存中gguf_type(int32_t)、value_len(uint64_t)、value   
    b. array在内存中GGUF_TYPE_ARRAY,gguf_type(int32_t)、value_len、value;gguf_type指数组内元素类型  

**tensor_info**   
- name_len(uint64) + name(string) 
- n_dims(4字节)  
- dims[4](uint64)、
- datatype(uint32)、  
- offset(uint64,偏移量32位对齐)  

**Tensor_Data**  
- offset : 读取完tensor_info数据后,ftell(fp)结果alignment对齐后为tensor数据部分的offset;  
- size :  single_tensor_size = sizeof(datatype) * np.prod(dims), 结果alignemnet字节对齐。遍历tensor累加得到size  

**gguf_context**  
head(version) + metadata + tensor_info + Tensor_Data.offset + Tensor_Data.size + alignment + data= gguf_context    


**gguf模型加载**    
```c
// 1. 根据model path 创建gguf_context
std::string model_fname = "model.gguf";
struct ggml_context * tmp_ctx = nullptr;
struct gguf_init_params gguf_params = {
    /*.no_alloc   =*/ false,
    /*.ctx        =*/ &tmp_ctx,
};
// no_alloc = false时会为tensor和weight分配内存，其中ggml_ctx.mem_buffer_owned = true
gguf_context * gguf_ctx = gguf_init_from_file(model_fname.c_str(), gguf_params);

// 2. 创建空gguf_context
gguf_context * gguf_ctx = gguf_init_empty();
```

**gguf_init_from_file_impl**  
1. 分配并初始化gguf_ctx  
2. 如果gguf_params.ctx!=nullptr,初始化ggml_ctx并分配内存池，用于存储模型tensor meta数据(gguf_params.no_alloc = false时额外分配tensor内存)。 

## 2.ggml_context 
- mem_size                      模型权重大小
- mem_buffer                    申请align(mem_size,64)大小的的buf
- mem_buffer_owned              buf所有权。是否ctx拥有，或者外部传递
- no_alloc                      是否禁止分配（用于共享内存模式）
- n_objects                     ctx拥有的object数量
- objects_begin,                容器链表头
- objects_end                   容器链表尾

ctx通过链表管理所有tensor;每个tensor_data持有两个head信息，ggml_object ggml_tensor  
model_data也视为ggml_object,并为ctx的头节点。+1 指 model_data对应的ggml_tensor  


**ggml_new_tensor**



**获取ggml_context**  
```c
// 1. 调用接口
// no_alloc = true : 只分配tensor metadata ; 
// no_alloc = false : 分配tensor metadata和tensor_data。并且tensor_data对应的object为ctx.objects_begin
bool no_alloc = false;
const size_t mem_size =
    no_alloc ?
    (n_tensors    )*ggml_tensor_overhead() :
    (n_tensors + 1)*ggml_tensor_overhead() + ctx->size;// +1对应tensor_data
struct ggml_init_params params = {
/*mem_size   =*/ mem_size,
/*mem_buffer =*/ nullptr,
/*no_alloc   =*/ no_alloc,
};
struct ggml_context* ctx = ggml_init(params); 
```

### 3.ggml_backend
在介绍后端之前，先介绍后端注册表  

**3.1 ggml_backend_reg**   
参考src/ggml-backend-reg.cpp  
- get_reg()返回全局注册表 , static ggml_backend_registry对象用于操作后端(reg load unload) 
- ggml_backend_registry定义register_backend() register_device()函数用于注册后端和设备
- ggml_backend_*_reg()函数返回 static ggml_backend_reg对象, iface实现不同。插入backends容器
- 通过backend遍历遍历后端设备返回static ggml_backend_device对象插入devices容器中(一种后端允许多个device，例如多卡gpu)

具体实现:
register_backend(ggml_backend_*_reg()) register_backend中调用register_device  
  
使用:  
ggml-backend-reg.cpp中定义了一些接口ggml_backend_reg_count \ ggml_backend_dev_count 等调用get_reg  
get_reg构造函数根据编译配置自动完成后端、设备注册  

***3.2 ggml_backend**  
ggml_backend根据已注册的backend进行初始化，并返回ggml_backend对象

***3.3 ggml_backend_cpu**  

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

**3.4 ggml_backend_buffer_type**
buffer_type可以理解为一个类型描述符（buffer descriptor）, 描述内存类型、内存对齐字节数和device上的如何alloc、free

**3.5 ggml_backend_buffer**
ggml中一切数据（context、dataset、weight、output…）都被存放在 buffer 中。ggml使用buffer进行集成承载不同的数据，实现多种后端（CPU、GPU）设备内存的统一管理。**ggml_backend_buffer**是实现不同类型数据在多种后端上进行统一的接口对象
- struct ggml_backend_buffer_i  iface; 后端对buffer进行操作的接口 
- ggml_backend_buffer_type_t    buft; buffer所属后端类型
- void * context; 后端buffer地址
- size_t size; 后端buffer大小 
- enum ggml_backend_buffer_usage usage; buffer用途，any通用/weight权重数据/compute计算数据


获取cpu后端的ggml_backend_buffer
```c
ggml_backend_t backend = ggml_backend_cpu_init();//backend
struct ggml_init_params pdata = {  
    /*mem_size   =*/ mem_size,
    /*mem_buffer =*/ nullptr,
    /*no_alloc   =*/ params.no_alloc,
};

ggml_context** ggml_ctx= ggml_init(pdata); // ggml_context
// buftype + buftype.iface.alloc_buffer
ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(backend, ggml_ctx);
/* 
// 根据backend.device获取buftype，一个操作ggml_backend_buffer的分配器
ggml_backend_buffer_type_t buftptet  = ggml_backend_get_default_buffer_type(backend); 
// 遍历tensors ,调用buftype.iface.alloc_buffer申请buf
// 每个tensor对应的ggml_backend_buffer_t组合到一个ggml_backend_buffer_t对象中，并返回
ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ggml_ctx,buftptet) 
/*
```

### 4.ggml_cgraph
graph的构建有两个步骤
- 使用算子函数连接weight参数、创建中间计算节点  
- 使用ggml_build_forward_expand()函数构建计算图

以examples/gpt-2为例:



### 5.ggml_schedule
分配计算节点实际计算
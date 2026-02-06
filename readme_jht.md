# ggml

## ggml overview
gguf : ggml模型格式
ggml_context: ggml核心数据结构，定义tensors_meta、tensors_buf  
ggml_cgraph: 计算图的表示，定义后端op计算执行顺序
ggml_backend: graph不同硬件计算抽象表示,  包含计算与内存管理, CPU (默认) 、CUDA、Metal (Apple Silicon) 、Vulkan、RPC 等等
ggml_backend_buffer_type: 后端buf描述符, 对应 ggml_backend 的mem_Management
ggml_backend_buffer: buffer_type实际分配的buf。一个ggml_backend_buffer可以存储多个张量数据, ggml数据（context、dataset、weight、output…）都存放在 buffer 中
ggml_gallocr: 计算图的mem_Management，用于不同后端张量内存管理  
ggml_backend_sched: 后端调度器  

## 1. GGUF
gguf模型，解析后保存到gguf_context  
**1.1 GGUF文件结构**  
[ Header ] -> [ Metadata ] -> [ Tensor Info ] -> [ Tensor Data ]  
gguf-header   
- magic;           GGUF的ASCII码 0x46554747  
- version;         GGUF版本 目前是3  
- n_tensor;        tensor数量  
- n_kv;            metadata数量(kv对数量)    

metadata   
- gguf_metadata_kv_t metadata_kv[metadata_kv_count];      
key(string)  
value   
    a. NO_array元素在内存中gguf_type(int32_t)、value_len(uint64_t)、value   
    b. array在内存中GGUF_TYPE_ARRAY,gguf_type(int32_t)、value_len、value;gguf_type指数组内元素类型  

tensor_info   
- name_len(uint64) + name(string)   
- n_dims(4字节)  
- dims \[4\](uint64)  
- datatype(uint32)    
- offset(uint64,偏移量32位对齐)    

Tensor_Data 
- offset : 读取完tensor_info数据后,ftell(fp)结果alignment对齐后为tensor数据部分的offset;  
- size :  single_tensor_size = sizeof(datatype) * np.prod(dims), 结果alignemnet字节对齐。遍历tensor累加得到size  

**1.2 gguf_context**  
gguf_context = head(version) + metadata + tensor_info + Tensor_Data.offset + Tensor_Data.size + alignment + data    


**1.3 gguf模型加载**    
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

`gguf_init_from_file_impl`  
1. 分配并初始化struct gguf_ctx  
2. 如果gguf_params.ctx!=nullptr,初始化ggml_ctx并根据size分配内存池，用于存储模型tensor meta数据(gguf_params.no_alloc = false时额外分配tensor内存)。 

## 2. ggml_context  
ctx通过链表管理所有tensor;每个tensor_data持有两个head信息，ggml_object ggml_tensor  
model_data也视为ggml_object,并为ctx的头节点。+1 指 model_data对应的ggml_tensor  

**2.1 ggml_new_tensor**
为每个tensor在ggml_context内存池分配内存。分配内存包括ggml_object ggml_tensor tensor_data
1. 为ggml_object分配内存  
在ggml_context的内存池中分配object对象内存
```c
struct ggml_object {  
    size_t offs;   // struct ggml_object尾指针相比较于mem_buffer的偏移量
    size_t size;   //object大小(包含  struct ggml_tensor在内)
}
```
2. 为ggml_tensor分配内存  
struct ggml_object * const obj;  
struct ggml_tensor * const tensor_new = (char *)ctx->mem_buffer + obj->offs  
3. 为tensor_data分配内存  
if !no_alloc 。分配tensor_data内存

**2.2 ggml_context**
获取ggml_context  
```c
// 1. 调用接口
// no_alloc = true : 只分配tensor metadata ; 
// no_alloc = false : 分配tensor metadata和weight_data。并且weight_data对应的object为ctx.objects_begin
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
 
ggml_context的mem_buffer内存布局有两种:  
a.  gguf_init_params.no_alloc == false && gguf_init_params.ctx != nullptr; ->init ggml_ctx 
    gguf_init_from_file中。先为所有tensor_data分配整块内存，后续ggml_new_tensor，需set(no_alloc,true)时，只分配metadata内存。         Abcbcbcbcbc  
b.  gguf_init_params.ctx == nullptr; 独立初始化ggml_context 。tensor_num个[object_metadata tensor_metadata + tensor_data]内存中排列  abcabcabc...abc  

## 3.ggml_cgraph
graph需要自定义构建，流程:  
- 分配ggml_cgraph内存
- 定义计算图，使用算子函数连接weight参数、创建中间计算节点
- 使用ggml_build_forward_expand()函数构建计算图
- 默认最大节点数size = 2048  

node为计算节点，指中间运算结果 ，数组中的元素是指向tensor的指针  
leaf为权重、常量、输入等  
ggml_cgraph_eval_order代表递归时src数组遍历正序或逆序，影响拓扑排序  


**3.1 内存分配**  
内存分配分为图张量内存分配器ggml_galloc_t和图内存申请ggml_new_graph  
ggml_new_graph : 为计算图分配内存  
ggml_galloc_t : 为计算图中的张量分配实际内存  

`ggml_new_graph(ggml_ctx)`
- 根据size计算graph所需内存大小，包含node leaf hash
- ctx根据size新建graph_obj，插入ggml ctx链表
- 计算包含node leaf hash等在graph_obj.data中的起始地址
- 结果返回ggml_cgraph*


**3.2 定义计算节点**  
例如定义y = d( a * w + b)。初始化ax y的src和op_type  
```c
struct ggml_tensor* aw          = ggml_mul(ggml_ctx, a, w);
struct ggml_tensor* y           = ggml_add(ggml_ctx, ax, b);
struct ggml_tensor* res_tensor  = ggml_mul(ggml_ctx, y, d);
```

**3.3 构建计算图**  
ggml_visit_parents从存放结果张量为根节点，每个节点依赖的张量为子节点，根据计算关系逆序遍历计算图  
遍历到的节点分别拷贝到ggml_cgraph nodes和leafs  

ggml_visit_parents(cgraph, res_tensor)判断逻辑  
- 从res_tensor开始，遍历所有依赖张量,tensor.src[:], cgraph.order决定tensor.src[:]遍历正序/逆序。默认正序
- hash_set记录已访问张量，避免重复遍历
- leafs: tensor.op == NULL && !param
- leafs以外都为node

以gpt2为例,从layer0.kv缓存开始构建计算图
```c
// tensor: embd  position memory_k memory_c  wte wpe
// weight:   norm(ln_1_g,ln_1_b) atten(c_attn_attn_w,c_attn_attn_b)
// constant: norm(eps)
inpl = ggml_add(ctx,ggml_get_rows(ctx, wte, embd),ggml_get_rows(ctx, wpe, position));
normRes = ggml_add(ctx,ggml_mul(ctx,ggml_norm(ctx, inpL, eps);,ln_1_g),ln_1_b);
attenres = ggml_add(ctx,ggml_mul_mat(ctx,c_attn_attn_w,cur),c_attn_attn_b);
Kcur = ggml_view_2d(ctx, cur, n_embd, N, cur->nb[1], 1*sizeof(float)*n_embd);
// store key and value to memory
k = ggml_view_1d(ctx, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur, k));
//添加节点顺序
// leaf                             c_attn_attn_w  
// leaf                             wte 
// leaf                             embd 
// node                             ggml_get_rows(wte) res 
// leaf                             model/wpe 
// leaf                             position 
// node                             ggml_get_rows(wpe) res 
// node                             inpl 
// node                             normRes
// leaf                             model/h0/ln_1/g 
// node                             norm中mul结果tensor
// leaf                             model/h0/ln_1/b 
// node                             norm中add结果tensor 
// node                             atten中mul结果tensor 
// leaf                             c_attn_attn_b 
// node                             attenres 
// node                             Kcur
// leaf                             memory_k
// node                             K
// node                             Kcur
```

## 4.ggml_backend
ggml后端通过5层抽象架构设计，分别是
Backend Registry → Device → Backend  
Backend Registry → Device → Buffer Type → Buffer  

Backend Registry 管理多个 Backend Registration  
每个 Registration 包含一个或多个 Device  
Device 可初始化 Backend 实例  
Device 提供 Buffer Type  
Buffer Type 负责分配具体的 Buffer  

以cpu后端为例  
cpu后端注册属于顶层接口，cpu_reg.iface包含获取cpu设备attr查询（name 、count）以及获取cpu_device实例  
cpu设备属于中间层。cpu_reg.device.iface包含device attr查询、后端初始化、获取Buffer Type  
cpu后端属于底层，cpu_reg.device.iface.init_backend.iface包含图创建、执行、tensor计算同步等  
cpu_reg —>  cpu_device -> cpu_device_iface -> cpu_backend +  cpu_backend_buffer_type_t  

各实例关系:
- cpu_reg 通过 iface.get_device 获取 cpu_device 实例；
- cpu_device 包含 reg 字段指向 cpu_reg（设备归属于注册器） 
- cpu_device 通过 iface.init_backend 创建 cpu_backend 实例 
- cpu_backend 包含 device 字段指向 cpu_device（后端实例归属于设备）
- 双向引用, 下层结构通过指针反向引用上层，便于访问父级资源
具体关系查看docs/ggml_backend.jpg docs/ggml_backend.png

**4.0.1 ggml_backend_registry**   
后端通过注册机制加入全局注册表,由全局注册表 ggml_backend_registry 统一管理,允许运行时动态加载后端,也可静态定义。
```c
struct ggml_backend_registry {
    std::vector<ggml_backend_reg_entry> backends; // 后端容器
    std::vector<ggml_backend_dev_t> devices; // 设备容器
    void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {} //后端注册
    void register_device(ggml_backend_dev_t device) {} //设备注册，在register_backend中调用
    ggml_backend_reg_t load_backend(const std::wstring & path, bool silent) {} // 动态加载指定路径so后端库
    void unload_backend(ggml_backend_reg_t reg, bool silent) {} //卸载后端
};
```

获取注册表
get_reg()函数，返回静态全局注册表对象  
```
static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    return reg;
}
```

枚举后端reg ggml_backend_reg_count、ggml_backend_reg_get  
枚举dev     ggml_backend_dev_count、ggml_backend_dev_get  


**4.0.2 动态加载**
通过 ggml_backend_load 打开动态库，查找符号 ggml_backend_score（0 表示当前系统不支持该后端）和 ggml_backend_init，调用后者获取 ggml_backend_reg_t 并注册  

```c
#include "ggml-backend.h"
//加载单个so后端
ggml_backend_load(so_path)
//批量加载
ggml_backend_load_all_from_path(foler_path)
//自动加载
ggml_backend_load_all()
```

ggml_backend_load_all自动对指定路径[./ , exec_path]中的libggml-{backend_name}-*.so评分，获取best backend，so保留dlopen句柄  

其中动态库编译时需使用宏 GGML_BACKEND_DL_IMPL(backend_reg_fn) 导出 ggml_backend_init 符号  

**4.0.3 静态注册**  
各后端实现ggml_backend_*_reg()函数，返回static ggml_backend_reg对象  
ggml_backend_registry构造函数调用后端提供的 ggml_backend_<name>_reg() 函数后端、设备注册  
1. static ggml_backend_reg插入注册表backends容器  
2. register_device()，遍历后端设备返回static ggml_backend_device对象，查询并插入注册表devices容器。(一种后端允许多个device，例如多卡gpu)   

注意:
backends容器中存储后端reg对象和so dlopen句柄（ggml_backend_reg而非ggml_backend），devices容器中存储后端device对象
实际的后端通过device.iface.init_backend 初始化

**4.1 ggml_backend_device**
ggml_backend_device连接后端注册表与实际执行后端,device关联后端reg对象,且device指针,需要通过接口函数访问

```c
struct ggml_backend_device {  
    struct ggml_backend_device_i iface;  // 设备接口函数  
    ggml_backend_reg_t reg;              // 所属后端注册表  
    void * context;                      // 设备特定上下文  
};
```

以cpu device.iface为例  
```c
static const struct ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,  // 获取后端
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type, //获取buffer_type
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr, //传参buf size初始化内存池 ggml_backend_buffer
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,  // 检查op->op op->src是否后端支持
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,   // 检查buft是否后端支持
};
```

获取device ,上下层架构双向引用, 获取方式很多
```c
// 1.枚举获取 或根据idx 获取
size_t count = ggml_backend_dev_count();  
for (size_t i = 0; i < count; i++) {  
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);  
}
// 2.根据backend_name获取
const char* backend_name = "CPU";
ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name);
// 3.根据backend_type获取
ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
// 4.根据backend_reg_t获取
size_t dev_count = ggml_backend_reg_dev_count(reg);  
for (size_t i = 0; i < dev_count; i++) {  
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);  
}
```


**4.2 ggml_backend**  
ggml_backend 是 GGML 中用于抽象不同硬件执行后端的核心接口, ggml_backend 结构体包含 GUID、接口 ggml_backend_i、device引用与context  

GUID 唯一标识backend  
iface 后端接口  
device 关联设备，用于操作设备iface接口  
context 以cpuBackend.ctx为例，存储线程池、线程数、cpu上计算图中间数据buf以及size、abort回调等  

**4.2.1 ggml_backend_cpu**  
获取cpu backend  
```c
//方式1 直接调用接口
ggml_backend_t backend = ggml_backend_cpu_init();

//方式2 获取dev，根据dev初始化backend  
const char* backend_name = "CPU";
ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name); // 检查device是否存在，返回dev
if (dev == nullptr) {
    fprintf(stderr, "%s: ERROR: backend %s not found, available:\n", __func__, backend_name.c_str());
}
ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);  // device->iface.init_backend,实际调用ggml_backend_cpu_init接口 

// 方式3  动态加载
ggml_backend_load_all();
ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr); 
```

**4.3 ggml_backend_buffer_type**
buffer_type = buf描述符（buffer descriptor） = backend的内存分配器, 描述内存类型、内存对齐字节数和以及device上如何alloc、free  
iface完成内存对齐、内存分配、以及不同后端之间数据传输  
```c
struct ggml_backend_buffer_type {
    struct ggml_backend_buffer_type_i  iface; // 一组接口函数
    ggml_backend_dev_t device; // 关联的后端设备
    void * context;
};
```

每个buffer type 都有一组 ggml_backend_buffer_type_i 接口函数  
    get_name: 返回 buffer type 名称  
    alloc_buffer: 分配指定大小的缓冲区  
    get_alignment: 返回内存对齐要求  
    get_max_size: 返回最大可分配大小（可选）  
    get_alloc_size: 计算张量实际占用大小（可选）  
    is_host: 判断是否为主机内存（可选）  


获取buft
以cpu为例
```c
// backend -> buft  返回static struct ggml_backend_buffer_type
ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend); //backend->device->iface.get_buffer_type(device)
```


**4.4 ggml_backend_buffer**
实际存放数据的buf
```c
struct ggml_backend_buffer {
    struct ggml_backend_buffer_i  iface; //操作ggml_backend_buffer的一组接口函数
    ggml_backend_buffer_type_t    buft; 
    void * context; // buffer ptr
    size_t size; // buffer size
    enum ggml_backend_buffer_usage usage; //buffer用途： any通用/weight权重数据/compute计算数据
};
```

cpu后端根据ggml_ctx申请内存，并返回ggml_backend_buffer  
```c
// get cpu backend
ggml_backend_t backend = ggml_backend_cpu_init();
struct ggml_init_params pdata = {  
    /*mem_size   =*/ mem_size,
    /*mem_buffer =*/ nullptr,
    /*no_alloc   =*/ params.no_alloc,
};
// get ggml_context
ggml_context** ggml_ctx= ggml_init(pdata); 
// 1. get buftype + buftype.iface.alloc_buffer for all tensor
// 2. combine all ggml_backend_buffer_t to one ggml_backend_buffer_t
// 3. ggml_backend_buffer_t以及addr回流至tensor，并init tensor buf（if required)
ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(backend, ggml_ctx);
```


## 5.ggml_gallocr
global allocator用于不同后端workspace_buf内存管理。  
通过预先规划内存布局，最大化内存复用。计算节点和输入节点，采用不同的内存管理策略。其次，使用哈希表张量快速查找  

ggml_backend_alloc_ctx_tensors申请的是权重内存，ggml_gallocr申请每个后端上的计算内存workspace_buf  

ggml_gallocr的使用流程：ggml_gallocr_new -> ggml_gallocr_reserve -> ggml_gallocr_alloc_graph -> ggml_gallocr_get_buffer_size
- 整个流程从后端缓冲区类型出发，创建一个专门用于计算图内存管理的分配器gallocr (gallocr_new)
- 然后通过最坏情况的计算图来精确估计计算过程中将使用的内存，然后预留这部分内存 (reserve)
- 最后查询并输出整个分配结果(get_buffer_size)

主要涉及两个结构体ggml_gallocr、ggml_dyn_tallocr  
ggml_gallocr : 图内存分配器,管理所有buft以及  
ggml_dyn_tallocr : 动态张量分配器。相同buft共享一个dyn_tallocr  

内存复用：
- 无空闲内存 或 空闲内存块size不满足要求，mem_alloc_size.push_back(size)
- 存在空闲内存块且size满足要求(大于且最靠近)。切割free_block,剩余free部分作为新的free_block

**5.1 ggml_gallocr_new_n**
分配gallocr内存;分配gallocr.bufs gallocr.buffers gallocr.buf_tallocs内存，calloc初始化0
```c
// 获取一个gallocr 根据backend.device.iface.buft
ggml_gallocr_t allocr = ggml_gallocr_new(buft)
// 获取n_bufs个gallocr , buft数组
ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs)
```
**5.2 ggml_gallocr_reserve_n**  
根据graph规划计算节点所需最小内存，并分配。流程如下  
- 根据graph统计的node和leaf节点数构建初始化哈希表并init(hash_size =  1.25 * (n_nodes + n_leafs))  0.25冗余
- 初始化动态张量分配器ggml_dyn_tallocr  
- 对于leaf节点 直接分配内存，统计hash表中view次数以及作为输入n_children次数  
- 对于node节点 遍历两次，第一次统计子节点数量n_children和视图计数（n_views） ++  
    第二次直接分配内存并更新hash表，拓扑顺序分配内存：确保父节点在子节点之前分配 --  
- 对于view= 0 ,n_children=0. 内存块回归free内存池  

**5.3 ggml_gallocr_alloc_graph**  
根据reserve阶段的规划，为graph中张量设置 data 指针  
- 根据tensor_hashmap.buf_id获取buft的base_ptr  
- tensor_hashmap.offset + base_ptr = tensor.data  


## 6.ggml_schedule
后端调度，分配节点计算。后端分配器的核心是子图切分，整个graph划分成多个subgraph，每个子图分配一个后端,ggml_schedule也负责不同后端之间的数据流转
调用流程如下:
- 定义计算图时,op通过接口ggml_backend_sched_set_tensor_backend指定tensor的backend（optional）
- 创建调度器ggml_backend_sched_new,其中backends按照优先级顺序排列（从高到低且backends[-1] = cpu_backend）
- 遍历graph,切分subGraph,记录切分子图时需要拷贝的tensor( hv_tensor_copies )
- ggml_backend_sched_graph_compute(sched, graph)  

**ggml_backend_sched_new**  
创建调度器,核心结构体ggml_backend_sched。申请内存并init,其中hv_tensor_backend_ids默认初始值-1
几个参数的大小设置
1.hv_tensor_copies的size:  
单流水线处理 hash_set.size * n_backends  并行处理 hash_set.size * n_backends * n_copys
```c
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
```
由tensor_id_copy可知hv_tensor_copies的内存布局，以两个tensor(tensor_a tensor_b)、两个backend为例 (hash_set.size = 2, n_backends = 2)  

hv_tensor_copies内存布局(2 张量a b 、2 后端 cpu gpu)  
if parallel  
- size=16（2×2×4），按 [a_cpu_cp0~3,a_gpu_cp0~3,b_cpu_cp0~3,b_gpu_cp0~3] 排列  

else  
- size=4（2×2×1），按 [a_cpu,a_gpu,b_cpu,b_gpu] 排列  

2.context_buffer_size  
ctx_size的大小考虑最差情况下,所有subgraph的tensor_cp占用内存以及graph占用内存  
- graph_size * subGraph_max_input_num * 2 * sizeof(struct ggml_tensor)  ，n个node,切分n个subgraph,一个node最多6输入6输出  
- ggml_graph_overhead_custom(graph_size, false);  graph大小  

3.splits、splits_capacity  
初始默认16个子图。后续根据需要翻倍扩容  

**ggml_backend_sched_split_graph**  
切分子图流程如下:
- 遍历graph->leafs 绑定backend_id ->hv_tensor_backend_ids
- 遍历graph->nodes , 根据tensor的src中weight的backend，结果和权重相同backend_id
- 遍历graph->nodes , 根据backend_id切分sched.split 相同id同一个subgraph。当前subgraph记录前一个subgraph的lastnode由于数据传输splits->input
- split与split.input的backend不同时，插入cp_tensor(仅backend不同)

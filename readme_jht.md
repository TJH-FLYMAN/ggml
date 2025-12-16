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



cpu后端相关
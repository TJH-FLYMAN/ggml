#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

void print_tensor_info(struct ggml_tensor* tensor, const char* name) {
    printf("Tensor: %s\n", name);
    printf("  ne: [%ld, %ld, %ld, %ld]\n", tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    printf("  nb: [%zu, %zu, %zu, %zu]\n", tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
    printf("  Type: %s\n", ggml_type_name(tensor->type));
    // Check properties
    printf("  Properties:\n");
    printf("    Contiguous: %s\n", ggml_is_contiguous(tensor) ? "Yes" : "No");
    printf("    Transposed: %s\n", ggml_is_transposed(tensor) ? "Yes" : "No");
    printf("    Permuted: %s\n", ggml_is_permuted(tensor) ? "Yes" : "No");
    printf("\n");
}

int main () {
    int tensor_num = 10;
    int tensor_size = GGML_PAD(640* 640 * 3  * sizeof(float) * tensor_num , 1024);
    struct ggml_init_params params {
        /*.mem_size   =*/ tensor_num *  ggml_graph_overhead() + tensor_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    ggml_context * ctx = ggml_init(params);

    // 只声明矩阵，无实际数据
    // pytorch中shape NCHW(1,3,640,640)  fp32
    // PyTorch等框架从最外到最内表示维度, ggml从内到外表示维度 
    // ne = [640,640,3,1]
    ggml_tensor* img_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 640, 640, 3, 1);
    // nb[0] = sizeif(fp32) = 4  第1个640维度上idx+1 需要指针偏移4字节，一行上右移
    // nb[1] = sizeif(fp32) * ne[0] = 4 * 640 = 2560  第2个640维度上idx+1 需要指针偏移完第一个维度上所有元素。从当前行的idx移动到下一行的idx。
    // nb[2] = sizeif(fp32) * ne[0] * ne[1] = nb[1] * ne[1]  =  2560 * 640 = 1638400    channel +1的对应位置
    // nb[3] = sizeif(fp32) * ne[0] * ne[1] * ne[2] = nb[2] * ne[2] = 1,638,400 * 3 = 4915200  batch + 1的对应位置
    // nb[i] = nb[i-1] * ne[i]
    printf("pytorch shape NCHW(1,3,640,640) fp32\n");
    print_tensor_info(img_tensor, "Image Tensor");  
    
    
    // 以Q4_0量化为例
    // 在内存层面，32个元素的block为最小可寻址单位。32个int4组合在一起，加上16位的delta 值（block_size = struct block_q4_0）  所以ne[0] = 18
    // 32个元素一组，640/32= 20组。nb[1] = 18 * 20 = 360  type_size * ne / block_size 
    // nb[2:]计算规则同上 nb[i] = nb[i-1] * ne[i]
    struct ggml_tensor* q4_img_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, 640, 640, 3, 1);
    print_tensor_info(q4_img_tensor, "Q4_0 Quantized img tensor");

    struct ggml_tensor* permuted_img_tensor = ggml_permute(ctx, img_tensor, 3, 2, 1, 0);
    print_tensor_info(permuted_img_tensor, "Permuted img tensor");

    struct ggml_tensor* transposed_img_tensor = ggml_transpose(ctx, img_tensor);
    print_tensor_info(transposed_img_tensor, "Transposed img tensor");
    
    ggml_free(ctx);
    return 0;
}

#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define GPT2_MAX_NODES 4096

struct gpt_model {
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};
static bool load_model(std::string & fname, gpt_model & model) {

    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();
    }
    struct ggml_context * tmp_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &tmp_ctx,
    };
    gguf_context * gguf_ctx = gguf_init_from_file(fname.c_str(), gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }
    
    uint32_t ver =  gguf_get_version(gguf_ctx);
    size_t alignment = gguf_get_alignment(gguf_ctx);
    size_t data_offset = gguf_get_data_offset(gguf_ctx);
    int64_t n_kv =  gguf_get_n_kv(gguf_ctx);
    printf("gguf version = %u\n", ver);
    printf("gguf alignment = %zu\n", alignment);
    printf("gguf data offset = %zu\n", data_offset);
    printf("gguf n_kv = %ld\n", n_kv);

#if 0
    for(int64_t i = 0; i < n_kv; i++){
        std::string key = gguf_get_key(gguf_ctx, i);
        printf(" key[%ld] = %s\n", i, key.c_str());
        
        enum gguf_type type = gguf_get_kv_type(gguf_ctx, i);
        printf("  type: %s (%d)\n", gguf_type_name(type), type);
        
        if (gguf_get_kv_type(gguf_ctx, i) == GGUF_TYPE_ARRAY) {
            enum gguf_type arr_type = gguf_get_arr_type(gguf_ctx, i);
            size_t arr_size = gguf_get_arr_n(gguf_ctx, i);
            printf("  array type: %s, size: %zu\n", gguf_type_name(arr_type), arr_size);
            
            switch (arr_type) {
                case GGUF_TYPE_UINT8: {
                    for (size_t j = 0; j < arr_size && j < 10; j++) { // 只打印前10个元素
                        uint8_t val = ((uint8_t*)gguf_get_arr_data(gguf_ctx, i))[j];
                        printf("    [%zu]: %u\n", j, val);
                    }
                    if (arr_size > 10) printf("    ... (%zu more elements)\n", arr_size - 10);
                    break;
                }
                case GGUF_TYPE_INT8: {
                    for (size_t j = 0; j < arr_size && j < 10; j++) {
                        int8_t val = ((int8_t*)gguf_get_arr_data(gguf_ctx, i))[j];
                        printf("    [%zu]: %d\n", j, val);
                    }
                    if (arr_size > 10) printf("    ... (%zu more elements)\n", arr_size - 10);
                    break;
                }
                case GGUF_TYPE_UINT16: {
                    for (size_t j = 0; j < arr_size && j < 10; j++) {
                        uint16_t val = ((uint16_t*)gguf_get_arr_data(gguf_ctx, i))[j];
                        printf("    [%zu]: %u\n", j, val);
                    }
                    if (arr_size > 10) printf("    ... (%zu more elements)\n", arr_size - 10);
                    break;
                }
                case GGUF_TYPE_INT16: {
                    for (size_t j = 0; j < arr_size && j < 10; j++) {
                        int16_t val = ((int16_t*)gguf_get_arr_data(gguf_ctx, i))[j];
                        printf("    [%zu]: %d\n", j, val);
                    }
                    if (arr_size > 10) printf("    ... (%zu more elements)\n", arr_size - 10);
                    break;
                }
                case GGUF_TYPE_UINT32: {
                    for (size_t j = 0; j < arr_size && j < 10; j++) {
                        uint32_t val = ((uint32_t*)gguf_get_arr_data(gguf_ctx, i))[j];
                        printf("    [%zu]: %u\n", j, val);
                    }
                    if (arr_size > 10) printf("    ... (%zu more elements)\n", arr_size - 10);
                    break;
                }
                case GGUF_TYPE_INT32: {
                    for (size_t j = 0; j < arr_size && j < 10; j++) {
                        int32_t val = ((int32_t*)gguf_get_arr_data(gguf_ctx, i))[j];
                        printf("    [%zu]: %d\n", j, val);
                    }
                    if (arr_size > 10) printf("    ... (%zu more elements)\n", arr_size - 10);
                    break;
                }
                case GGUF_TYPE_FLOAT32: {
                    for (size_t j = 0; j < arr_size && j < 10; j++) {
                        float val = ((float*)gguf_get_arr_data(gguf_ctx, i))[j];
                        printf("    [%zu]: %.6f\n", j, val);
                    }
                    if (arr_size > 10) printf("    ... (%zu more elements)\n", arr_size - 10);
                    break;
                }
                case GGUF_TYPE_STRING: {
                    for (size_t j = 0; j < arr_size && j < 10; j++) {
                        const char* val = gguf_get_arr_str(gguf_ctx, i, j);
                        printf("    [%zu]: %s\n", j, val);
                    }
                    if (arr_size > 10) printf("    ... (%zu more elements)\n", arr_size - 10);
                    break;
                }
                case GGUF_TYPE_BOOL: {
                    for (size_t j = 0; j < arr_size && j < 10; j++) {
                        bool val = ((bool*)gguf_get_arr_data(gguf_ctx, i))[j];
                        printf("    [%zu]: %s\n", j, val ? "true" : "false");
                    }
                    if (arr_size > 10) printf("    ... (%zu more elements)\n", arr_size - 10);
                    break;
                }
                default:
                    printf("    Unsupported array type: %d\n", arr_type);
                    break;
            }
        } else {

            switch (type) {
                case GGUF_TYPE_UINT8:
                    printf("  value: %u\n", gguf_get_val_u8(gguf_ctx, i));
                    break;
                case GGUF_TYPE_INT8:
                    printf("  value: %d\n", gguf_get_val_i8(gguf_ctx, i));
                    break;
                case GGUF_TYPE_UINT16:
                    printf("  value: %u\n", gguf_get_val_u16(gguf_ctx, i));
                    break;
                case GGUF_TYPE_INT16:
                    printf("  value: %d\n", gguf_get_val_i16(gguf_ctx, i));
                    break;
                case GGUF_TYPE_UINT32:
                    printf("  value: %u\n", gguf_get_val_u32(gguf_ctx, i));
                    break;
                case GGUF_TYPE_INT32:
                    printf("  value: %d\n", gguf_get_val_i32(gguf_ctx, i));
                    break;
                case GGUF_TYPE_FLOAT32:
                    printf("  value: %.6f\n", gguf_get_val_f32(gguf_ctx, i));
                    break;
                case GGUF_TYPE_STRING: {
                    const char* val = gguf_get_val_str(gguf_ctx, i);
                    printf("  value: %s\n", val);
                    break;
                }
                case GGUF_TYPE_BOOL:
                    printf("  value: %s\n", gguf_get_val_bool(gguf_ctx, i) ? "true" : "false");
                    break;
                case GGUF_TYPE_UINT64:
                    printf("  value: %lu\n", gguf_get_val_u64(gguf_ctx, i));
                    break;
                case GGUF_TYPE_INT64:
                    printf("  value: %ld\n", gguf_get_val_i64(gguf_ctx, i));
                    break;
                case GGUF_TYPE_FLOAT64:
                    printf("  value: %.6lf\n", gguf_get_val_f64(gguf_ctx, i));
                    break;
                default:
                    printf("  Unsupported type: %d\n", type);
                    break;
            }
        }
        printf("\n");
    }
#endif

    int num_tensors = gguf_get_n_tensors(gguf_ctx);
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };
    model.ctx = ggml_init(params);
    for (int i = 0; i < num_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
        struct ggml_tensor * dst = ggml_dup_tensor(model.ctx, src);
        ggml_set_name(dst, name);
    }

    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    // copy tensors from main memory to backend
    for (struct ggml_tensor * cur = ggml_get_first_tensor(model.ctx); cur != NULL; cur = ggml_get_next_tensor(model.ctx, cur)) {
        struct ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        size_t n_size = ggml_nbytes(src);
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, n_size);
    }
    
    // ggml_show_object(model.ctx);

    gguf_free(gguf_ctx);
    return true;
}

int main(int argc, char* argv[]){

    std::string fname = "/home/jhtang3/ggml/ggml/build/models/gpt-2-117M/ggml-model-f16.gguf";
    gpt_model model;

    ggml_time_init();

    bool rev = load_model(fname,model);

    return 1;

    rev = 1;
}
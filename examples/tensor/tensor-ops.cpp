#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace {

const char * tensor_name(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return "NULL";
    }
    return tensor->name[0] == '\0' ? "<unnamed>" : tensor->name;
}

void set_values(ggml_tensor * tensor, std::initializer_list<float> values) {
    GGML_ASSERT(static_cast<int64_t>(values.size()) == ggml_nelements(tensor));

    int index = 0;
    for (float value : values) {
        ggml_set_f32_1d(tensor, index++, value);
    }
}

ggml_tensor * new_vector(
        ggml_context * ctx,
        const char * name,
        std::initializer_list<float> values) {
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, values.size());
    ggml_set_name(tensor, name);
    set_values(tensor, values);
    return tensor;
}

void compute(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 1);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed: %s\n", ggml_status_to_string(status));
        std::abort();
    }
}

void print_shape(const ggml_tensor * tensor) {
    std::printf("[");
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        std::printf("%s%lld", i == 0 ? "" : ", ",
                static_cast<long long>(tensor->ne[i]));
    }
    std::printf("]");
}

void print_strides(const ggml_tensor * tensor) {
    std::printf("[");
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        std::printf("%s%zu", i == 0 ? "" : ", ", tensor->nb[i]);
    }
    std::printf("]");
}

void print_values(const char * label, const ggml_tensor * tensor) {
    std::printf("  %s = [", label);
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
        std::printf("%s%.2f", i == 0 ? "" : ", ", ggml_get_f32_1d(tensor, i));
    }
    std::printf("]\n");
}

void print_sources(const ggml_tensor * tensor) {
    std::printf("[");
    bool first = true;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (tensor->src[i] == nullptr) {
            continue;
        }
        std::printf("%s%s", first ? "" : ", ", tensor_name(tensor->src[i]));
        first = false;
    }
    std::printf("]");
}

// 只打印由本次操作改变的关键 metadata。普通逐元素操作的 ne/nb 不变，
// 因此只会显示新的 op 和 src；视图类操作还会显示 ne/nb/view/layout 的变化。
void print_metadata_changes(const ggml_tensor * input, const ggml_tensor * output) {
    std::printf("  op: %s -> %s\n", ggml_op_desc(input), ggml_op_desc(output));
    std::printf("  src: ");
    print_sources(output);
    std::printf("\n");

    if (std::memcmp(input->ne, output->ne, sizeof(input->ne)) != 0) {
        std::printf("  ne: ");
        print_shape(input);
        std::printf(" -> ");
        print_shape(output);
        std::printf("\n");
    }

    if (std::memcmp(input->nb, output->nb, sizeof(input->nb)) != 0) {
        std::printf("  nb: ");
        print_strides(input);
        std::printf(" -> ");
        print_strides(output);
        std::printf("\n");
    }

    if (input->view_src != output->view_src) {
        std::printf("  view_src: %s -> %s\n",
                tensor_name(input->view_src), tensor_name(output->view_src));
    }

    if (input->view_offs != output->view_offs) {
        std::printf("  view_offs: %zu -> %zu bytes\n",
                input->view_offs, output->view_offs);
    }

    if (ggml_is_contiguous(input) != ggml_is_contiguous(output)) {
        std::printf("  contiguous: %s -> %s\n",
                ggml_is_contiguous(input) ? "yes" : "no",
                ggml_is_contiguous(output) ? "yes" : "no");
    }
}

void print_title(const char * operation) {
    std::printf("\n== %s ==\n", operation);
}

// 新建 tensor 时最重要的是形状 ne 和字节步长 nb。
void example_tensor_creation(ggml_context * ctx) {
    print_title("tensor creation");

    ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 3, 4);
    ggml_set_name(tensor, "tensor_2x3x4");

    std::printf("  ne = ");
    print_shape(tensor);
    std::printf("\n  nb = ");
    print_strides(tensor);
    std::printf("\n  elements = %lld, bytes = %zu\n",
            static_cast<long long>(ggml_nelements(tensor)), ggml_nbytes(tensor));
}

void example_add(ggml_context * ctx) {
    print_title("ggml_add");

    ggml_tensor * a = new_vector(ctx, "a", {1, 2, 3});
    ggml_tensor * b = new_vector(ctx, "b", {10, 20, 30});
    ggml_tensor * output = ggml_add(ctx, a, b);
    ggml_set_name(output, "add_out");

    print_metadata_changes(a, output);
    compute(ctx, output);
    print_values("a", a);
    print_values("b", b);
    print_values("output", output);
}

void example_sub(ggml_context * ctx) {
    print_title("ggml_sub");

    ggml_tensor * a = new_vector(ctx, "a", {5, 7, 9});
    ggml_tensor * b = new_vector(ctx, "b", {1, 2, 3});
    ggml_tensor * output = ggml_sub(ctx, a, b);
    ggml_set_name(output, "sub_out");

    print_metadata_changes(a, output);
    compute(ctx, output);
    print_values("a", a);
    print_values("b", b);
    print_values("output", output);
}

void example_mul(ggml_context * ctx) {
    print_title("ggml_mul");

    ggml_tensor * a = new_vector(ctx, "a", {1, 2, 3});
    ggml_tensor * b = new_vector(ctx, "b", {2, 3, 4});
    ggml_tensor * output = ggml_mul(ctx, a, b);
    ggml_set_name(output, "mul_out");

    print_metadata_changes(a, output);
    compute(ctx, output);
    print_values("a", a);
    print_values("b", b);
    print_values("output", output);
}

void example_div(ggml_context * ctx) {
    print_title("ggml_div");

    ggml_tensor * a = new_vector(ctx, "a", {2, 6, 12});
    ggml_tensor * b = new_vector(ctx, "b", {2, 3, 4});
    ggml_tensor * output = ggml_div(ctx, a, b);
    ggml_set_name(output, "div_out");

    print_metadata_changes(a, output);
    compute(ctx, output);
    print_values("a", a);
    print_values("b", b);
    print_values("output", output);
}

void example_scale(ggml_context * ctx) {
    print_title("ggml_scale");

    ggml_tensor * input = new_vector(ctx, "input", {1, 2, 3});
    ggml_tensor * output = ggml_scale(ctx, input, 2.0f);
    ggml_set_name(output, "scale_out");

    print_metadata_changes(input, output);
    std::printf("  scale: 2.00\n");
    compute(ctx, output);
    print_values("input", input);
    print_values("output", output);
}

void example_relu(ggml_context * ctx) {
    print_title("ggml_relu");

    ggml_tensor * input = new_vector(ctx, "input", {-2, 0, 3});
    ggml_tensor * output = ggml_relu(ctx, input);
    ggml_set_name(output, "relu_out");

    print_metadata_changes(input, output);
    compute(ctx, output);
    print_values("input", input);
    print_values("output", output);
}

void example_sqr(ggml_context * ctx) {
    print_title("ggml_sqr");

    ggml_tensor * input = new_vector(ctx, "input", {-2, 3, 4});
    ggml_tensor * output = ggml_sqr(ctx, input);
    ggml_set_name(output, "sqr_out");

    print_metadata_changes(input, output);
    compute(ctx, output);
    print_values("input", input);
    print_values("output", output);
}

void example_sqrt(ggml_context * ctx) {
    print_title("ggml_sqrt");

    ggml_tensor * input = new_vector(ctx, "input", {1, 4, 9});
    ggml_tensor * output = ggml_sqrt(ctx, input);
    ggml_set_name(output, "sqrt_out");

    print_metadata_changes(input, output);
    compute(ctx, output);
    print_values("input", input);
    print_values("output", output);
}

void example_repeat(ggml_context * ctx) {
    print_title("ggml_repeat");

    ggml_tensor * row = new_vector(ctx, "row", {10, 20, 30});
    ggml_tensor * target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_set_name(target, "target_3x2");
    ggml_tensor * output = ggml_repeat(ctx, row, target);
    ggml_set_name(output, "repeat_out");

    // target 只提供目标形状，不是计算图依赖，所以 output->src 中只有 row。
    print_metadata_changes(row, output);
    compute(ctx, output);
    print_values("row", row);
    print_values("output", output);
}

void example_broadcast_add(ggml_context * ctx) {
    print_title("ggml_add (broadcast)");

    ggml_tensor * matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_set_name(matrix, "matrix_3x2");
    set_values(matrix, {1, 2, 3, 4, 5, 6});
    ggml_tensor * row = new_vector(ctx, "row", {10, 20, 30});
    ggml_tensor * output = ggml_add(ctx, matrix, row);
    ggml_set_name(output, "broadcast_add_out");

    // row 在 ne[1] 方向自动重复，output 的形状仍与 matrix 相同。
    print_metadata_changes(matrix, output);
    compute(ctx, output);
    print_values("matrix", matrix);
    print_values("row", row);
    print_values("output", output);
}

void example_sum(ggml_context * ctx) {
    print_title("ggml_sum");

    ggml_tensor * input = new_vector(ctx, "input", {1, 2, 3, 4});
    ggml_tensor * output = ggml_sum(ctx, input);
    ggml_set_name(output, "sum_out");

    print_metadata_changes(input, output);
    compute(ctx, output);
    print_values("input", input);
    print_values("output", output);
}

void example_sum_rows(ggml_context * ctx) {
    print_title("ggml_sum_rows");

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_set_name(input, "matrix_3x2");
    set_values(input, {1, 2, 3, 4, 5, 6});
    ggml_tensor * output = ggml_sum_rows(ctx, input);
    ggml_set_name(output, "sum_rows_out");

    // ne[0] 是一行的元素数；sum_rows 将每行压缩成一个值。
    print_metadata_changes(input, output);
    compute(ctx, output);
    print_values("input", input);
    print_values("output", output);
}

void example_mean(ggml_context * ctx) {
    print_title("ggml_mean");

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_set_name(input, "matrix_3x2");
    set_values(input, {1, 2, 3, 4, 5, 6});
    ggml_tensor * output = ggml_mean(ctx, input);
    ggml_set_name(output, "mean_out");

    // ggml_mean 与 sum_rows 一样沿 ne[0] 归约，但计算的是均值。
    print_metadata_changes(input, output);
    compute(ctx, output);
    print_values("input", input);
    print_values("output", output);
}

void example_concat(ggml_context * ctx) {
    print_title("ggml_concat");

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    ggml_set_name(a, "a_2x2");
    ggml_set_name(b, "b_2x2");
    set_values(a, {1, 2, 3, 4});
    set_values(b, {5, 6, 7, 8});
    ggml_tensor * output = ggml_concat(ctx, a, b, 0);
    ggml_set_name(output, "concat_out");

    print_metadata_changes(a, output);
    std::printf("  concat dimension: 0\n");
    compute(ctx, output);
    print_values("a", a);
    print_values("b", b);
    print_values("output", output);
}

void example_reshape(ggml_context * ctx) {
    print_title("ggml_reshape_2d");

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_set_name(input, "matrix_4x3");
    set_values(input, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    ggml_tensor * output = ggml_reshape_2d(ctx, input, 2, 6);
    ggml_set_name(output, "reshape_out");

    // reshape 只改 ne/nb，元素总数和底层存储不变。
    print_metadata_changes(input, output);
    print_values("output", output);
}

void example_view(ggml_context * ctx) {
    print_title("ggml_view_2d");

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_set_name(input, "matrix_4x3");
    set_values(input, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    // 每行从第 2 个元素开始取 2 个元素；最后一个参数是 1 个 float 的偏移量。
    ggml_tensor * output = ggml_view_2d(
            ctx, input, 2, 3, input->nb[1], sizeof(float));
    ggml_set_name(output, "view_out");

    print_metadata_changes(input, output);
    print_values("output", output);
}

void example_transpose(ggml_context * ctx) {
    print_title("ggml_transpose");

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_set_name(input, "matrix_4x3");
    set_values(input, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    ggml_tensor * output = ggml_transpose(ctx, input);
    ggml_set_name(output, "transpose_out");

    // transpose 交换 ne[0]/ne[1] 和 nb[0]/nb[1]，返回共享存储的非连续视图。
    print_metadata_changes(input, output);
    print_values("output", output);
}

void example_permute(ggml_context * ctx) {
    print_title("ggml_permute");

    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 3, 2);
    ggml_set_name(input, "tensor_2x3x2");
    set_values(input, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    ggml_tensor * output = ggml_permute(ctx, input, 1, 0, 2, 3);
    ggml_set_name(output, "permute_out");

    // 将前两个轴交换；和 transpose 一样只改 metadata，不复制数据。
    print_metadata_changes(input, output);
    std::printf("  axes: [1, 0, 2, 3]\n");
    print_values("output", output);
}

void example_cont(ggml_context * ctx) {
    print_title("ggml_cont");

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_set_name(input, "matrix_4x3");
    set_values(input, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    ggml_tensor * transposed = ggml_transpose(ctx, input);
    ggml_set_name(transposed, "transpose_view");
    ggml_tensor * output = ggml_cont(ctx, transposed);
    ggml_set_name(output, "cont_out");

    // ggml_cont 将非连续视图物化为独立的连续存储。
    print_metadata_changes(transposed, output);
    compute(ctx, output);
    print_values("output", output);
}

void example_mul_mat(ggml_context * ctx) {
    print_title("ggml_mul_mat");

    // ne[0] 是每行元素数。这里计算 a * b^T，输出形状为 [a.ne[1], b.ne[1]]。
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_set_name(a, "a_3x2");
    ggml_set_name(b, "b_3x2");
    set_values(a, {1, 2, 3, 4, 5, 6});
    set_values(b, {7, 8, 9, 10, 11, 12});
    ggml_tensor * output = ggml_mul_mat(ctx, a, b);
    ggml_set_name(output, "mul_mat_out");

    print_metadata_changes(a, output);
    compute(ctx, output);
    print_values("output", output);
}

} // namespace

int main() {
    // legacy context 同时管理 tensor、计算图和数据；这些小样例共用一个 context 即可。
    const ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create ggml context\n");
        return 1;
    }

    example_tensor_creation(ctx);
    example_add(ctx);
    example_sub(ctx);
    example_mul(ctx);
    example_div(ctx);
    example_scale(ctx);
    example_relu(ctx);
    example_sqr(ctx);
    example_sqrt(ctx);
    example_repeat(ctx);
    example_broadcast_add(ctx);
    example_sum(ctx);
    example_sum_rows(ctx);
    example_mean(ctx);
    example_concat(ctx);
    example_reshape(ctx);
    example_view(ctx);
    example_transpose(ctx);
    example_permute(ctx);
    example_cont(ctx);
    example_mul_mat(ctx);

    ggml_free(ctx);
    return 0;
}

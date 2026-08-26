#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <initializer_list>
#include <string>

namespace {

constexpr float EPSILON = 1e-5f;

struct test_runner {
    int passed = 0;
    int failed = 0;

    void expect(bool condition, const std::string & name, const std::string & detail = "") {
        if (condition) {
            ++passed;
            std::printf("[PASS] %s\n", name.c_str());
            return;
        }

        ++failed;
        std::fprintf(stderr, "[FAIL] %s", name.c_str());
        if (!detail.empty()) {
            std::fprintf(stderr, ": %s", detail.c_str());
        }
        std::fprintf(stderr, "\n");
    }
};

// ggml_xxx() 调用会创建一个新的输出 tensor，并立即填写下面这些 metadata；
// 真正的数值计算要等到 ggml_graph_compute_with_ctx()。这个快照用于验证：
// 1. out-of-place 操作没有修改输入 tensor 的 metadata；
// 2. graph compute 只填写 data 指向的存储，不会再次修改 tensor metadata。
struct tensor_metadata_snapshot {
    ggml_type type;
    const void * buffer;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    ggml_op op;
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t flags;
    const ggml_tensor * src[GGML_MAX_SRC];
    const ggml_tensor * view_src;
    size_t view_offs;
    const void * data;
    const void * extra;
};

tensor_metadata_snapshot snapshot_metadata(const ggml_tensor * tensor) {
    tensor_metadata_snapshot result = {};
    result.type = tensor->type;
    result.buffer = tensor->buffer;
    result.op = tensor->op;
    result.flags = tensor->flags;
    result.view_src = tensor->view_src;
    result.view_offs = tensor->view_offs;
    result.data = tensor->data;
    result.extra = tensor->extra;

    std::memcpy(result.ne, tensor->ne, sizeof(result.ne));
    std::memcpy(result.nb, tensor->nb, sizeof(result.nb));
    std::memcpy(result.op_params, tensor->op_params, sizeof(result.op_params));
    std::memcpy(result.src, tensor->src, sizeof(result.src));
    return result;
}

bool metadata_equals(const tensor_metadata_snapshot & before, const ggml_tensor * after) {
    return before.type == after->type &&
           before.buffer == after->buffer &&
           std::memcmp(before.ne, after->ne, sizeof(before.ne)) == 0 &&
           std::memcmp(before.nb, after->nb, sizeof(before.nb)) == 0 &&
           before.op == after->op &&
           std::memcmp(before.op_params, after->op_params, sizeof(before.op_params)) == 0 &&
           before.flags == after->flags &&
           std::memcmp(before.src, after->src, sizeof(before.src)) == 0 &&
           before.view_src == after->view_src &&
           before.view_offs == after->view_offs &&
           before.data == after->data &&
           before.extra == after->extra;
}

const char * tensor_name(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return "NULL";
    }
    return tensor->name[0] == '\0' ? "<unnamed>" : tensor->name;
}

bool same_ne(const ggml_tensor * lhs, const ggml_tensor * rhs) {
    return std::memcmp(lhs->ne, rhs->ne, sizeof(lhs->ne)) == 0;
}

bool same_nb(const ggml_tensor * lhs, const ggml_tensor * rhs) {
    return std::memcmp(lhs->nb, rhs->nb, sizeof(lhs->nb)) == 0;
}

bool same_src(const ggml_tensor * lhs, const ggml_tensor * rhs) {
    return std::memcmp(lhs->src, rhs->src, sizeof(lhs->src)) == 0;
}

bool same_op_params(const ggml_tensor * lhs, const ggml_tensor * rhs) {
    return std::memcmp(lhs->op_params, rhs->op_params, sizeof(lhs->op_params)) == 0;
}

const char * change_mark(bool unchanged) {
    return unchanged ? "unchanged" : "CHANGED";
}

void print_src_list(const ggml_tensor * tensor) {
    std::printf("[");
    bool first = true;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (tensor->src[i] == nullptr) {
            continue;
        }
        std::printf("%s%d:%s@%p",
                first ? "" : ", ",
                i,
                tensor_name(tensor->src[i]),
                static_cast<void *>(tensor->src[i]));
        first = false;
    }
    std::printf("]");
}

void print_view_source(const ggml_tensor * tensor) {
    if (tensor->view_src == nullptr) {
        std::printf("NULL");
        return;
    }
    std::printf("%s@%p", tensor_name(tensor->view_src), static_cast<void *>(tensor->view_src));
}

void print_operation_parameter_meaning(const ggml_tensor * tensor) {
    switch (tensor->op) {
        case GGML_OP_SCALE: {
            float scale = 0.0f;
            std::memcpy(&scale, tensor->op_params, sizeof(scale));
            std::printf("  %-12s           scale=%.3f\n", "params_as", scale);
        } break;
        case GGML_OP_UNARY:
            std::printf("  %-12s           unary=%s\n", "params_as",
                    ggml_unary_op_name(ggml_get_unary_op(tensor)));
            break;
        case GGML_OP_CONCAT:
            std::printf("  %-12s           concat_dim=%d\n", "params_as", tensor->op_params[0]);
            break;
        case GGML_OP_VIEW: {
            size_t offset = 0;
            std::memcpy(&offset, tensor->op_params, sizeof(offset));
            std::printf("  %-12s           view_offset=%zu bytes\n", "params_as", offset);
        } break;
        case GGML_OP_PERMUTE:
            std::printf("  %-12s           axes=[%d, %d, %d, %d]\n", "params_as",
                    tensor->op_params[0], tensor->op_params[1],
                    tensor->op_params[2], tensor->op_params[3]);
            break;
        default:
            break;
    }
}

// 对比输入和输出 tensor。CHANGED 表示该 metadata 字段和左侧 tensor 不同，
// 并不表示输入 tensor 被原地修改。绝大多数 ggml 操作都是创建新的输出 tensor。
void report_metadata_changes(
        const char * operation,
        const ggml_tensor * before,
        const ggml_tensor * after) {
    const bool type_same = before->type == after->type;
    const bool ne_same = same_ne(before, after);
    const bool nb_same = same_nb(before, after);
    const bool op_same = before->op == after->op;
    const bool op_params_same = same_op_params(before, after);
    const bool flags_same = before->flags == after->flags;
    const bool src_same = same_src(before, after);
    const bool view_src_same = before->view_src == after->view_src;
    const bool view_offs_same = before->view_offs == after->view_offs;
    const bool data_same = before->data == after->data;
    const bool buffer_same = before->buffer == after->buffer;
    const bool elements_same = ggml_nelements(before) == ggml_nelements(after);
    const bool bytes_same = ggml_nbytes(before) == ggml_nbytes(after);
    const bool contiguous_same = ggml_is_contiguous(before) == ggml_is_contiguous(after);
    const bool transposed_same = ggml_is_transposed(before) == ggml_is_transposed(after);
    const bool permuted_same = ggml_is_permuted(before) == ggml_is_permuted(after);

    std::printf("\n[META] %s: %s -> %s\n", operation, tensor_name(before), tensor_name(after));
    std::printf("  %-12s %-9s %s -> %s\n", "type", change_mark(type_same),
            ggml_type_name(before->type), ggml_type_name(after->type));
    std::printf("  %-12s %-9s [%lld, %lld, %lld, %lld] -> [%lld, %lld, %lld, %lld]\n",
            "ne", change_mark(ne_same),
            static_cast<long long>(before->ne[0]), static_cast<long long>(before->ne[1]),
            static_cast<long long>(before->ne[2]), static_cast<long long>(before->ne[3]),
            static_cast<long long>(after->ne[0]), static_cast<long long>(after->ne[1]),
            static_cast<long long>(after->ne[2]), static_cast<long long>(after->ne[3]));
    std::printf("  %-12s %-9s [%zu, %zu, %zu, %zu] -> [%zu, %zu, %zu, %zu]\n",
            "nb", change_mark(nb_same),
            before->nb[0], before->nb[1], before->nb[2], before->nb[3],
            after->nb[0], after->nb[1], after->nb[2], after->nb[3]);
    std::printf("  %-12s %-9s %s -> %s\n", "op", change_mark(op_same),
            ggml_op_name(before->op), ggml_op_name(after->op));
    std::printf("  %-12s %-9s [%d, %d, %d, %d] -> [%d, %d, %d, %d]\n",
            "op_params", change_mark(op_params_same),
            before->op_params[0], before->op_params[1], before->op_params[2], before->op_params[3],
            after->op_params[0], after->op_params[1], after->op_params[2], after->op_params[3]);
    print_operation_parameter_meaning(after);
    std::printf("  %-12s %-9s 0x%x -> 0x%x\n", "flags", change_mark(flags_same),
            before->flags, after->flags);
    std::printf("  %-12s %-9s ", "src[]", change_mark(src_same));
    print_src_list(before);
    std::printf(" -> ");
    print_src_list(after);
    std::printf("\n");
    std::printf("  %-12s %-9s ", "view_src", change_mark(view_src_same));
    print_view_source(before);
    std::printf(" -> ");
    print_view_source(after);
    std::printf("\n");
    std::printf("  %-12s %-9s %zu -> %zu bytes\n", "view_offs", change_mark(view_offs_same),
            before->view_offs, after->view_offs);
    std::printf("  %-12s %-9s %p -> %p%s\n", "data", change_mark(data_same),
            before->data, after->data,
            after->view_src != nullptr ? " (shares storage through view_src)" : " (independent output storage)");
    std::printf("  %-12s %-9s %p -> %p\n", "buffer", change_mark(buffer_same),
            static_cast<void *>(before->buffer), static_cast<void *>(after->buffer));
    std::printf("  %-12s %-9s %lld -> %lld\n", "n_elements", change_mark(elements_same),
            static_cast<long long>(ggml_nelements(before)),
            static_cast<long long>(ggml_nelements(after)));
    std::printf("  %-12s %-9s %zu -> %zu\n", "nbytes", change_mark(bytes_same),
            ggml_nbytes(before), ggml_nbytes(after));
    std::printf("  %-12s %-9s contiguous=%d/transposed=%d/permuted=%d"
                " -> contiguous=%d/transposed=%d/permuted=%d\n",
            "layout", change_mark(contiguous_same && transposed_same && permuted_same),
            ggml_is_contiguous(before), ggml_is_transposed(before), ggml_is_permuted(before),
            ggml_is_contiguous(after), ggml_is_transposed(after), ggml_is_permuted(after));
}

void expect_operation_metadata(
        test_runner & runner,
        const ggml_tensor * result,
        ggml_op expected_op,
        const ggml_tensor * expected_src0,
        const ggml_tensor * expected_src1,
        const std::string & name) {
    bool sources_match = result->src[0] == expected_src0 && result->src[1] == expected_src1;
    for (int i = 2; i < GGML_MAX_SRC; ++i) {
        sources_match = sources_match && result->src[i] == nullptr;
    }
    runner.expect(result->op == expected_op && sources_match, name,
            "unexpected op or src[] dependency metadata");
}

void expect_metadata_unchanged(
        test_runner & runner,
        const tensor_metadata_snapshot & before,
        const ggml_tensor * after,
        const std::string & name) {
    runner.expect(metadata_equals(before, after), name, "metadata changed unexpectedly");
}

void print_metadata_legend() {
    std::printf(
            "GGML tensor metadata fields:\n"
            "  type       : element data type\n"
            "  ne[d]      : number of elements on dimension d (GGML stores inner dimension first)\n"
            "  nb[d]      : byte stride for moving one position on dimension d\n"
            "  op         : operation that produces this tensor\n"
            "  op_params  : operation-specific parameters (shown as the first four int32 slots)\n"
            "  flags      : INPUT / OUTPUT / PARAM / LOSS flags\n"
            "  src[]      : input dependencies used when the graph computes this tensor\n"
            "  view_src   : owner of shared storage for reshape/view/transpose/permute\n"
            "  view_offs  : byte offset into view_src storage\n"
            "  data       : address of this tensor's first logical element\n"
            "  buffer     : backend buffer owner (NULL with the legacy context used here)\n"
            "  layout     : derived contiguous/transposed/permuted properties\n"
            "\n"
            "Each [META] block compares an input tensor with the NEW output tensor.\n"
            "CHANGED does not mean that the input was modified in-place. Numeric computation\n"
            "writes output data later and is separately checked not to alter metadata.\n");
}

void set_values(ggml_tensor * tensor, std::initializer_list<float> values) {
    GGML_ASSERT(static_cast<int64_t>(values.size()) == ggml_nelements(tensor));

    int64_t index = 0;
    for (float value : values) {
        ggml_set_f32_1d(tensor, index++, value);
    }
}

void expect_values(
        test_runner & runner,
        const ggml_tensor * tensor,
        std::initializer_list<float> expected,
        const std::string & name,
        float tolerance = EPSILON) {
    if (static_cast<int64_t>(expected.size()) != ggml_nelements(tensor)) {
        runner.expect(false, name, "element count does not match");
        return;
    }

    int64_t index = 0;
    for (float expected_value : expected) {
        const float actual_value = ggml_get_f32_1d(tensor, index);
        if (std::fabs(actual_value - expected_value) > tolerance) {
            char detail[160];
            std::snprintf(
                    detail,
                    sizeof(detail),
                    "element %lld: expected %.6f, got %.6f",
                    static_cast<long long>(index),
                    expected_value,
                    actual_value);
            runner.expect(false, name, detail);
            return;
        }
        ++index;
    }

    runner.expect(true, name);
}

void expect_shape(
        test_runner & runner,
        const ggml_tensor * tensor,
        std::initializer_list<int64_t> expected,
        const std::string & name) {
    int dimension = 0;
    for (int64_t expected_size : expected) {
        if (dimension >= GGML_MAX_DIMS || tensor->ne[dimension] != expected_size) {
            char detail[160];
            std::snprintf(
                    detail,
                    sizeof(detail),
                    "dimension %d: expected %lld, got %lld",
                    dimension,
                    static_cast<long long>(expected_size),
                    dimension < GGML_MAX_DIMS ? static_cast<long long>(tensor->ne[dimension]) : -1LL);
            runner.expect(false, name, detail);
            return;
        }
        ++dimension;
    }

    for (; dimension < GGML_MAX_DIMS; ++dimension) {
        if (tensor->ne[dimension] != 1) {
            runner.expect(false, name, "an unspecified trailing dimension is not 1");
            return;
        }
    }

    runner.expect(true, name);
}

bool compute(
        ggml_context * ctx,
        test_runner & runner,
        const std::string & name,
        std::initializer_list<ggml_tensor *> outputs) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    for (ggml_tensor * output : outputs) {
        ggml_build_forward_expand(graph, output);
    }

    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 1);
    const bool success = status == GGML_STATUS_SUCCESS;
    runner.expect(success, name, success ? "" : ggml_status_to_string(status));
    return success;
}

void test_tensor_metadata(ggml_context * ctx, test_runner & runner) {
    std::printf("\n== tensor creation and metadata ==\n");

    ggml_tensor * tensor_1d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
    ggml_tensor * tensor_2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_tensor * tensor_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 3, 4);
    ggml_tensor * tensor_4d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 3, 4, 5);

    expect_shape(runner, tensor_1d, {5}, "create 1D tensor [5]");
    expect_shape(runner, tensor_2d, {3, 2}, "create 2D tensor [3, 2]");
    expect_shape(runner, tensor_3d, {2, 3, 4}, "create 3D tensor [2, 3, 4]");
    expect_shape(runner, tensor_4d, {2, 3, 4, 5}, "create 4D tensor [2, 3, 4, 5]");

    runner.expect(ggml_nelements(tensor_4d) == 120, "element count");
    runner.expect(ggml_nbytes(tensor_2d) == 6 * sizeof(float), "tensor byte size");
    runner.expect(
            tensor_2d->nb[0] == sizeof(float) && tensor_2d->nb[1] == 3 * sizeof(float),
            "contiguous strides (nb)");
    runner.expect(ggml_is_contiguous(tensor_4d), "new tensor is contiguous");
}

void test_elementwise_operations(ggml_context * ctx, test_runner & runner) {
    std::printf("\n== element-wise and activation operations ==\n");

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_set_name(a, "elementwise_a");
    ggml_set_name(b, "elementwise_b");
    set_values(a, {-3.0f, -2.0f, -1.0f, 1.0f, 2.0f, 3.0f});
    set_values(b, { 1.0f,  2.0f,  4.0f, 2.0f, 4.0f, 6.0f});
    const tensor_metadata_snapshot a_before = snapshot_metadata(a);

    ggml_tensor * add       = ggml_add(ctx, a, b);
    ggml_tensor * subtract  = ggml_sub(ctx, a, b);
    ggml_tensor * multiply  = ggml_mul(ctx, a, b);
    ggml_tensor * divide    = ggml_div(ctx, a, b);
    ggml_tensor * scaled    = ggml_scale(ctx, a, 2.0f);
    ggml_tensor * relu      = ggml_relu(ctx, a);
    ggml_tensor * squared   = ggml_sqr(ctx, a);
    ggml_tensor * abs_value = ggml_sqrt(ctx, squared);

    ggml_set_name(add, "add_out");
    ggml_set_name(subtract, "sub_out");
    ggml_set_name(multiply, "mul_out");
    ggml_set_name(divide, "div_out");
    ggml_set_name(scaled, "scale_out");
    ggml_set_name(relu, "relu_out");
    ggml_set_name(squared, "sqr_out");
    ggml_set_name(abs_value, "sqrt_out");

    report_metadata_changes("ggml_add", a, add);
    report_metadata_changes("ggml_sub", a, subtract);
    report_metadata_changes("ggml_mul", a, multiply);
    report_metadata_changes("ggml_div", a, divide);
    report_metadata_changes("ggml_scale", a, scaled);
    report_metadata_changes("ggml_relu", a, relu);
    report_metadata_changes("ggml_sqr", a, squared);
    report_metadata_changes("ggml_sqrt", squared, abs_value);

    expect_operation_metadata(runner, add, GGML_OP_ADD, a, b, "add op/src metadata");
    expect_operation_metadata(runner, subtract, GGML_OP_SUB, a, b, "sub op/src metadata");
    expect_operation_metadata(runner, multiply, GGML_OP_MUL, a, b, "mul op/src metadata");
    expect_operation_metadata(runner, divide, GGML_OP_DIV, a, b, "div op/src metadata");
    expect_operation_metadata(runner, scaled, GGML_OP_SCALE, a, nullptr, "scale op/src metadata");
    expect_operation_metadata(runner, relu, GGML_OP_UNARY, a, nullptr, "ReLU op/src metadata");
    expect_operation_metadata(runner, squared, GGML_OP_SQR, a, nullptr, "sqr op/src metadata");
    expect_operation_metadata(runner, abs_value, GGML_OP_SQRT, squared, nullptr, "sqrt op/src metadata");
    const tensor_metadata_snapshot add_before_compute = snapshot_metadata(add);

    if (!compute(ctx, runner, "compute element-wise graph", {
                add, subtract, multiply, divide, scaled, relu, squared, abs_value})) {
        return;
    }

    expect_metadata_unchanged(runner, a_before, a,
            "out-of-place operations keep input metadata");
    expect_metadata_unchanged(runner, add_before_compute, add,
            "graph compute keeps output metadata");

    expect_values(runner, add,       {-2,  0,  3,  3,  6,  9}, "add");
    expect_values(runner, subtract,  {-4, -4, -5, -1, -2, -3}, "subtract");
    expect_values(runner, multiply,  {-3, -4, -4,  2,  8, 18}, "multiply");
    expect_values(runner, divide,    {-3, -1, -0.25f, 0.5f, 0.5f, 0.5f}, "divide");
    expect_values(runner, scaled,    {-6, -4, -2,  2,  4,  6}, "scale");
    expect_values(runner, relu,      { 0,  0,  0,  1,  2,  3}, "ReLU");
    expect_values(runner, squared,   { 9,  4,  1,  1,  4,  9}, "square");
    expect_values(runner, abs_value, { 3,  2,  1,  1,  2,  3}, "square + square root");
}

void test_broadcast_reduce_and_concat(ggml_context * ctx, test_runner & runner) {
    std::printf("\n== broadcast, reduction and concatenation ==\n");

    ggml_tensor * row = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    ggml_tensor * matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_set_name(row, "broadcast_row");
    ggml_set_name(matrix, "matrix_3x2");
    set_values(row, {10, 20, 30});
    set_values(matrix, {1, 2, 3, 4, 5, 6});
    const tensor_metadata_snapshot matrix_before = snapshot_metadata(matrix);

    ggml_tensor * repeated = ggml_repeat(ctx, row, matrix);
    ggml_tensor * broadcast_add = ggml_add(ctx, matrix, row);
    ggml_tensor * sum = ggml_sum(ctx, matrix);
    ggml_tensor * row_sums = ggml_sum_rows(ctx, matrix);
    ggml_tensor * row_means = ggml_mean(ctx, matrix);
    ggml_set_name(repeated, "repeat_out");
    ggml_set_name(broadcast_add, "broadcast_add_out");
    ggml_set_name(sum, "sum_out");
    ggml_set_name(row_sums, "sum_rows_out");
    ggml_set_name(row_means, "mean_rows_out");

    ggml_tensor * concat_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    ggml_tensor * concat_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    ggml_set_name(concat_a, "concat_a");
    ggml_set_name(concat_b, "concat_b");
    set_values(concat_a, {1, 2, 3, 4});
    set_values(concat_b, {5, 6, 7, 8});
    ggml_tensor * concatenated = ggml_concat(ctx, concat_a, concat_b, 0);
    ggml_set_name(concatenated, "concat_out");

    report_metadata_changes("ggml_repeat", row, repeated);
    report_metadata_changes("ggml_add (broadcast)", matrix, broadcast_add);
    report_metadata_changes("ggml_sum", matrix, sum);
    report_metadata_changes("ggml_sum_rows", matrix, row_sums);
    report_metadata_changes("ggml_mean", matrix, row_means);
    report_metadata_changes("ggml_concat(dim=0)", concat_a, concatenated);

    expect_operation_metadata(runner, repeated, GGML_OP_REPEAT, row, nullptr,
            "repeat op/src metadata");
    expect_operation_metadata(runner, broadcast_add, GGML_OP_ADD, matrix, row,
            "broadcast add op/src metadata");
    expect_operation_metadata(runner, sum, GGML_OP_SUM, matrix, nullptr,
            "sum op/src metadata");
    expect_operation_metadata(runner, row_sums, GGML_OP_SUM_ROWS, matrix, nullptr,
            "sum_rows op/src metadata");
    expect_operation_metadata(runner, row_means, GGML_OP_MEAN, matrix, nullptr,
            "mean op/src metadata");
    expect_operation_metadata(runner, concatenated, GGML_OP_CONCAT, concat_a, concat_b,
            "concat op/src metadata");
    const tensor_metadata_snapshot sum_before_compute = snapshot_metadata(sum);

    if (!compute(ctx, runner, "compute broadcast/reduction graph", {
                repeated, broadcast_add, sum, row_sums, row_means, concatenated})) {
        return;
    }

    expect_metadata_unchanged(runner, matrix_before, matrix,
            "broadcast/reduction operations keep input metadata");
    expect_metadata_unchanged(runner, sum_before_compute, sum,
            "reduction compute keeps output metadata");

    expect_values(runner, repeated,     {10, 20, 30, 10, 20, 30}, "repeat row");
    expect_values(runner, broadcast_add, {11, 22, 33, 14, 25, 36}, "broadcast add");
    expect_values(runner, sum,           {21}, "sum all elements");
    expect_values(runner, row_sums,      {6, 15}, "sum rows");
    expect_values(runner, row_means,     {2, 5}, "mean rows");
    expect_shape(runner, concatenated, {4, 2}, "concat shape");
    expect_values(runner, concatenated, {1, 2, 5, 6, 3, 4, 7, 8}, "concat on dimension 0");
}

void test_shape_and_view_operations(ggml_context * ctx, test_runner & runner) {
    std::printf("\n== reshape, view, transpose and permute ==\n");

    ggml_tensor * matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_set_name(matrix, "shape_matrix_4x3");
    set_values(matrix, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    const tensor_metadata_snapshot matrix_before = snapshot_metadata(matrix);

    // reshape 只修改张量元数据；元素总数必须保持不变。
    ggml_tensor * reshaped = ggml_reshape_2d(ctx, matrix, 2, 6);
    ggml_set_name(reshaped, "reshape_out");

    // 从每行第 2 个元素开始，取连续的 2 个元素。
    ggml_tensor * view = ggml_view_2d(
            ctx, matrix, 2, 3, matrix->nb[1], matrix->nb[0]);
    ggml_set_name(view, "view_out");
    ggml_tensor * contiguous_view = ggml_cont(ctx, view);
    ggml_set_name(contiguous_view, "view_cont_out");

    // transpose/permute 返回的是改变步长的视图，ggml_cont 将其物化为连续内存。
    ggml_tensor * transposed = ggml_transpose(ctx, matrix);
    ggml_set_name(transposed, "transpose_out");
    ggml_tensor * contiguous_transpose = ggml_cont(ctx, transposed);
    ggml_set_name(contiguous_transpose, "transpose_cont_out");

    ggml_tensor * tensor_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 3, 2);
    ggml_set_name(tensor_3d, "permute_input_2x3x2");
    set_values(tensor_3d, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    ggml_tensor * permuted = ggml_permute(ctx, tensor_3d, 1, 0, 2, 3);
    ggml_set_name(permuted, "permute_out");
    ggml_tensor * contiguous_permute = ggml_cont(ctx, permuted);
    ggml_set_name(contiguous_permute, "permute_cont_out");

    report_metadata_changes("ggml_reshape_2d", matrix, reshaped);
    report_metadata_changes("ggml_view_2d", matrix, view);
    report_metadata_changes("ggml_cont(view)", view, contiguous_view);
    report_metadata_changes("ggml_transpose", matrix, transposed);
    report_metadata_changes("ggml_cont(transpose)", transposed, contiguous_transpose);
    report_metadata_changes("ggml_permute", tensor_3d, permuted);
    report_metadata_changes("ggml_cont(permute)", permuted, contiguous_permute);

    expect_operation_metadata(runner, reshaped, GGML_OP_RESHAPE, matrix, nullptr,
            "reshape op/src metadata");
    expect_operation_metadata(runner, view, GGML_OP_VIEW, matrix, nullptr,
            "view op/src metadata");
    expect_operation_metadata(runner, contiguous_view, GGML_OP_CONT, view, nullptr,
            "cont(view) op/src metadata");
    expect_operation_metadata(runner, transposed, GGML_OP_TRANSPOSE, matrix, nullptr,
            "transpose op/src metadata");
    expect_operation_metadata(runner, contiguous_transpose, GGML_OP_CONT, transposed, nullptr,
            "cont(transpose) op/src metadata");
    expect_operation_metadata(runner, permuted, GGML_OP_PERMUTE, tensor_3d, nullptr,
            "permute op/src metadata");
    expect_operation_metadata(runner, contiguous_permute, GGML_OP_CONT, permuted, nullptr,
            "cont(permute) op/src metadata");

    runner.expect(reshaped->view_src == matrix && reshaped->data == matrix->data &&
                    !same_ne(reshaped, matrix) && !same_nb(reshaped, matrix),
            "reshape changes ne/nb but shares input storage");
    runner.expect(view->view_src == matrix && view->view_offs == sizeof(float) &&
                    view->data == static_cast<char *>(matrix->data) + sizeof(float),
            "view records view_src/view_offs and offsets data pointer");
    runner.expect(transposed->view_src == matrix && transposed->data == matrix->data &&
                    transposed->ne[0] == matrix->ne[1] && transposed->nb[0] == matrix->nb[1],
            "transpose swaps ne/nb and shares input storage");
    runner.expect(permuted->view_src == tensor_3d && permuted->data == tensor_3d->data &&
                    !same_ne(permuted, tensor_3d) && !same_nb(permuted, tensor_3d),
            "permute reorders ne/nb and shares input storage");
    runner.expect(contiguous_transpose->view_src == nullptr &&
                    contiguous_transpose->data != transposed->data &&
                    ggml_is_contiguous(contiguous_transpose),
            "cont creates independent contiguous storage");
    const tensor_metadata_snapshot transpose_cont_before_compute =
            snapshot_metadata(contiguous_transpose);

    expect_shape(runner, reshaped, {2, 6}, "reshape [4, 3] to [2, 6]");
    expect_values(runner, reshaped, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, "reshape keeps data order");
    runner.expect(!ggml_is_contiguous(transposed), "transpose is a non-contiguous view");

    if (!compute(ctx, runner, "compute view/transpose/permute graph", {
                contiguous_view, contiguous_transpose, contiguous_permute})) {
        return;
    }

    expect_metadata_unchanged(runner, matrix_before, matrix,
            "shape/view operations keep input metadata");
    expect_metadata_unchanged(runner, transpose_cont_before_compute, contiguous_transpose,
            "layout compute keeps output metadata");

    expect_shape(runner, contiguous_view, {2, 3}, "view shape");
    expect_values(runner, contiguous_view, {2, 3, 6, 7, 10, 11}, "strided 2D view");
    expect_shape(runner, contiguous_transpose, {3, 4}, "transpose shape");
    expect_values(runner, contiguous_transpose,
            {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12}, "transpose values");
    expect_shape(runner, contiguous_permute, {3, 2, 2}, "permute shape");
    expect_values(runner, contiguous_permute,
            {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12}, "permute axes");
}

void test_matrix_multiplication(ggml_context * ctx, test_runner & runner) {
    std::printf("\n== matrix multiplication ==\n");

    // GGML 的 ne[0] 是一行的元素数。mul_mat(a, b) 会计算 a * b^T。
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_set_name(a, "matmul_a");
    ggml_set_name(b, "matmul_b");
    set_values(a, {1, 2, 3, 4, 5, 6});
    set_values(b, {7, 8, 9, 10, 11, 12});
    const tensor_metadata_snapshot a_before = snapshot_metadata(a);

    ggml_tensor * result = ggml_mul_mat(ctx, a, b);
    ggml_set_name(result, "matmul_out");
    report_metadata_changes("ggml_mul_mat", a, result);
    expect_operation_metadata(runner, result, GGML_OP_MUL_MAT, a, b,
            "mul_mat op/src metadata");
    runner.expect(result->ne[0] == a->ne[1] && result->ne[1] == b->ne[1] &&
                    result->type == GGML_TYPE_F32 && ggml_is_contiguous(result),
            "mul_mat derives output ne/nb/type metadata");
    const tensor_metadata_snapshot result_before_compute = snapshot_metadata(result);

    if (!compute(ctx, runner, "compute matrix multiplication graph", {result})) {
        return;
    }

    expect_metadata_unchanged(runner, a_before, a,
            "mul_mat keeps input metadata");
    expect_metadata_unchanged(runner, result_before_compute, result,
            "mul_mat compute keeps output metadata");

    expect_shape(runner, result, {2, 2}, "matrix multiplication shape");
    expect_values(runner, result, {50, 122, 68, 167}, "matrix multiplication values");
}

} // namespace

int main() {
    ggml_time_init();
    print_metadata_legend();

    // 本例使用 legacy context API，让输入、输出和计算图都由同一个 context 管理。
    const size_t context_size = 16 * 1024 * 1024;
    const ggml_init_params params = {
        /* .mem_size   = */ context_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create ggml context\n");
        return 1;
    }

    test_runner runner;
    test_tensor_metadata(ctx, runner);
    test_elementwise_operations(ctx, runner);
    test_broadcast_reduce_and_concat(ctx, runner);
    test_shape_and_view_operations(ctx, runner);
    test_matrix_multiplication(ctx, runner);

    std::printf("\nSummary: %d passed, %d failed\n", runner.passed, runner.failed);
    ggml_free(ctx);

    return runner.failed == 0 ? 0 : 1;
}

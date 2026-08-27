#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char * DEFAULT_MODEL_PATH =
        "models/gpt2-Q4KM/gpt2.Q4_K_M.gguf";
constexpr size_t VALUE_PREVIEW_COUNT = 8;
constexpr size_t STRING_PREVIEW_BYTES = 80;

struct tensor_info {
    std::string name;
    enum ggml_type type = GGML_TYPE_COUNT;
    int n_dims = 0;
    std::array<int64_t, GGML_MAX_DIMS> ne {{ 1, 1, 1, 1 }};
    uint64_t file_offset = 0;
    size_t nbytes = 0;
};

struct model_info {
    std::string path;
    uint64_t file_size = 0;
    uint64_t data_offset = 0;
    std::vector<tensor_info> tensors;
};

struct legacy_gpt2_info {
    int32_t n_vocab = 0;
    int32_t n_ctx   = 0;
    int32_t n_embd  = 0;
    int32_t n_head  = 0;
    int32_t n_layer = 0;
    int32_t ftype   = 0;
    std::vector<std::string> vocab;
};

static bool add_size(size_t & total, size_t value) {
    if (value > std::numeric_limits<size_t>::max() - total) {
        return false;
    }
    total += value;
    return true;
}

static bool mul_size(size_t & total, size_t value) {
    if (value != 0 && total > std::numeric_limits<size_t>::max()/value) {
        return false;
    }
    total *= value;
    return true;
}

static size_t pad_to(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static std::string format_bytes(uint64_t bytes) {
    char buffer[96];
    const char * unit = "B";
    double value = static_cast<double>(bytes);
    if (bytes >= 1024ULL*1024ULL*1024ULL) {
        value /= 1024.0*1024.0*1024.0;
        unit = "GiB";
    } else if (bytes >= 1024ULL*1024ULL) {
        value /= 1024.0*1024.0;
        unit = "MiB";
    } else if (bytes >= 1024ULL) {
        value /= 1024.0;
        unit = "KiB";
    }
    std::snprintf(buffer, sizeof(buffer), "%.2f %s (%" PRIu64 " B)", value, unit, bytes);
    return buffer;
}

static std::string escape_string(const std::string & value, size_t max_bytes = STRING_PREVIEW_BYTES) {
    static const char hex[] = "0123456789abcdef";
    std::string result;
    const size_t n = std::min(value.size(), max_bytes);
    for (size_t i = 0; i < n; ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        switch (c) {
            case '\\': result += "\\\\"; break;
            case '"':  result += "\\\""; break;
            case '\n': result += "\\n";  break;
            case '\r': result += "\\r";  break;
            case '\t': result += "\\t";  break;
            default:
                if (c >= 0x20 && c <= 0x7e) {
                    result += static_cast<char>(c);
                } else {
                    result += "\\x";
                    result += hex[c >> 4];
                    result += hex[c & 0x0f];
                }
                break;
        }
    }
    if (value.size() > n) {
        result += "...";
    }
    return result;
}

static std::string format_shape(const tensor_info & info) {
    std::string result = "[";
    for (int d = 0; d < info.n_dims; ++d) {
        if (d != 0) {
            result += " x ";
        }
        result += std::to_string(info.ne[d]);
    }
    result += "]";
    return result;
}

template <typename T>
static bool read_value(std::ifstream & input, T & value) {
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    return input.good();
}

static bool get_file_size(std::ifstream & input, uint64_t & size) {
    input.clear();
    input.seekg(0, std::ios::end);
    const std::streampos end = input.tellg();
    if (end < 0) {
        return false;
    }
    size = static_cast<uint64_t>(end);
    input.seekg(0, std::ios::beg);
    return input.good();
}

static bool compute_tensor_nbytes(tensor_info & info, std::string & error) {
    if (info.type < 0 || info.type >= GGML_TYPE_COUNT) {
        error = "invalid ggml tensor type " + std::to_string(static_cast<int>(info.type));
        return false;
    }

    const int64_t block_size = ggml_blck_size(info.type);
    if (block_size <= 0 || info.ne[0] <= 0 || info.ne[0] % block_size != 0) {
        error = "ne[0] is not compatible with the tensor block size";
        return false;
    }

    size_t nbytes = ggml_row_size(info.type, info.ne[0]);
    for (int d = 1; d < info.n_dims; ++d) {
        if (info.ne[d] <= 0 || !mul_size(nbytes, static_cast<size_t>(info.ne[d]))) {
            error = "tensor byte size overflows size_t";
            return false;
        }
    }
    info.nbytes = nbytes;
    return true;
}

static bool inspect_magic(const std::string & path, std::array<char, 4> & magic, uint64_t & file_size) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "error: cannot open '%s'\n", path.c_str());
        return false;
    }
    if (!get_file_size(input, file_size) || file_size < magic.size()) {
        std::fprintf(stderr, "error: '%s' is too small to be a model file\n", path.c_str());
        return false;
    }
    input.read(magic.data(), magic.size());
    if (!input) {
        std::fprintf(stderr, "error: cannot read the magic from '%s'\n", path.c_str());
        return false;
    }
    return true;
}

static void print_tensor_table(const model_info & model) {
    std::printf("\n  tensor directory\n");
    std::printf("  %-4s %-44s %-8s %-22s %-13s %s\n",
            "id", "name", "type", "shape (ggml order)", "bytes", "absolute file offset");
    std::printf("  %s\n", std::string(116, '-').c_str());

    uint64_t total = 0;
    for (size_t i = 0; i < model.tensors.size(); ++i) {
        const tensor_info & info = model.tensors[i];
        total += info.nbytes;
        std::printf("  %-4zu %-44.44s %-8s %-22.22s %-13zu 0x%012" PRIx64 "\n",
                i, info.name.c_str(), ggml_type_name(info.type), format_shape(info).c_str(),
                info.nbytes, info.file_offset);
    }
    std::printf("\n  tensor payload total: %s\n", format_bytes(total).c_str());
}

static bool parse_legacy_ggml(
        const std::string & path,
        model_info & model,
        legacy_gpt2_info & legacy) {
    std::ifstream input(path, std::ios::binary);
    if (!input || !get_file_size(input, model.file_size)) {
        std::fprintf(stderr, "error: cannot open legacy GGML file '%s'\n", path.c_str());
        return false;
    }

    uint32_t magic = 0;
    if (!read_value(input, magic) || magic != GGML_FILE_MAGIC ||
        !read_value(input, legacy.n_vocab) || !read_value(input, legacy.n_ctx) ||
        !read_value(input, legacy.n_embd)  || !read_value(input, legacy.n_head) ||
        !read_value(input, legacy.n_layer) || !read_value(input, legacy.ftype)) {
        std::fprintf(stderr, "error: invalid or truncated legacy GPT-2 header\n");
        return false;
    }

    if (legacy.n_vocab <= 0 || legacy.n_ctx <= 0 || legacy.n_embd <= 0 ||
        legacy.n_head <= 0 || legacy.n_layer <= 0) {
        std::fprintf(stderr, "error: invalid legacy GPT-2 hyperparameters\n");
        return false;
    }

    int32_t vocab_size = 0;
    if (!read_value(input, vocab_size) || vocab_size != legacy.n_vocab) {
        std::fprintf(stderr, "error: invalid legacy vocabulary size\n");
        return false;
    }

    legacy.vocab.reserve(static_cast<size_t>(vocab_size));
    for (int32_t i = 0; i < vocab_size; ++i) {
        uint32_t length = 0;
        if (!read_value(input, length) || length > model.file_size) {
            std::fprintf(stderr, "error: invalid vocabulary entry at index %d\n", i);
            return false;
        }
        std::string token(length, '\0');
        if (length != 0) {
            input.read(&token[0], length);
        }
        if (!input) {
            std::fprintf(stderr, "error: truncated vocabulary entry at index %d\n", i);
            return false;
        }
        legacy.vocab.push_back(std::move(token));
    }

    model.data_offset = static_cast<uint64_t>(input.tellg());
    while (true) {
        const int next = input.peek();
        if (next == std::char_traits<char>::eof()) {
            break;
        }

        int32_t n_dims = 0;
        int32_t name_length = 0;
        int32_t type_value = 0;
        if (!read_value(input, n_dims) || !read_value(input, name_length) || !read_value(input, type_value)) {
            std::fprintf(stderr, "error: truncated legacy tensor header\n");
            return false;
        }
        if (n_dims < 1 || n_dims > GGML_MAX_DIMS || name_length <= 0 || name_length >= GGML_MAX_NAME ||
            type_value < 0 || type_value >= GGML_TYPE_COUNT) {
            std::fprintf(stderr,
                    "error: invalid legacy tensor header (n_dims=%d, name_length=%d, type=%d)\n",
                    n_dims, name_length, type_value);
            return false;
        }

        tensor_info info;
        info.n_dims = n_dims;
        info.type = static_cast<enum ggml_type>(type_value);
        for (int d = 0; d < n_dims; ++d) {
            int32_t ne = 0;
            if (!read_value(input, ne) || ne <= 0) {
                std::fprintf(stderr, "error: invalid dimension in legacy tensor %zu\n", model.tensors.size());
                return false;
            }
            info.ne[d] = ne;
        }

        info.name.resize(static_cast<size_t>(name_length));
        input.read(&info.name[0], name_length);
        if (!input) {
            std::fprintf(stderr, "error: truncated legacy tensor name\n");
            return false;
        }

        std::string size_error;
        if (!compute_tensor_nbytes(info, size_error)) {
            std::fprintf(stderr, "error: tensor '%s': %s\n", info.name.c_str(), size_error.c_str());
            return false;
        }

        const std::streampos data_pos = input.tellg();
        if (data_pos < 0) {
            std::fprintf(stderr, "error: cannot determine data offset for tensor '%s'\n", info.name.c_str());
            return false;
        }
        info.file_offset = static_cast<uint64_t>(data_pos);
        if (info.file_offset > model.file_size || info.nbytes > model.file_size - info.file_offset) {
            std::fprintf(stderr, "error: truncated data for tensor '%s'\n", info.name.c_str());
            return false;
        }

        input.seekg(static_cast<std::streamoff>(info.nbytes), std::ios::cur);
        if (!input) {
            std::fprintf(stderr, "error: cannot skip data for tensor '%s'\n", info.name.c_str());
            return false;
        }
        model.tensors.push_back(std::move(info));
    }

    if (model.tensors.empty()) {
        std::fprintf(stderr, "error: the legacy model contains no tensors\n");
        return false;
    }
    return true;
}

static void visualize_legacy_file(const model_info & model, const legacy_gpt2_info & legacy) {
    std::printf("\n========== 1. MODEL FILE VISUALIZATION ==========\n");
    std::printf("\n  NOTE: this input is legacy GGML, not GGUF.\n");
    std::printf("  magic bytes are 'lmgg' (little-endian GGML_FILE_MAGIC); GGUF must start with 'GGUF'.\n\n");
    std::printf("  %s\n", model.path.c_str());
    std::printf("  [legacy header: 28 B]\n");
    std::printf("       |-- n_vocab = %d\n", legacy.n_vocab);
    std::printf("       |-- n_ctx   = %d\n", legacy.n_ctx);
    std::printf("       |-- n_embd  = %d\n", legacy.n_embd);
    std::printf("       |-- n_head  = %d\n", legacy.n_head);
    std::printf("       |-- n_layer = %d\n", legacy.n_layer);
    std::printf("       `-- ftype   = %d\n", legacy.ftype);
    std::printf("  [vocabulary: %zu entries]\n", legacy.vocab.size());
    for (size_t i = 0; i < std::min(VALUE_PREVIEW_COUNT, legacy.vocab.size()); ++i) {
        std::printf("       |-- token[%zu] = \"%s\"\n", i, escape_string(legacy.vocab[i]).c_str());
    }
    if (legacy.vocab.size() > VALUE_PREVIEW_COUNT) {
        const size_t last = legacy.vocab.size() - 1;
        std::printf("       |-- ... %zu entries omitted ...\n", legacy.vocab.size() - VALUE_PREVIEW_COUNT - 1);
        std::printf("       `-- token[%zu] = \"%s\"\n", last, escape_string(legacy.vocab[last]).c_str());
    }
    std::printf("  [tensor records begin at 0x%012" PRIx64 "]\n", model.data_offset);
    std::printf("  [file end: 0x%012" PRIx64 ", %s]\n",
            model.file_size, format_bytes(model.file_size).c_str());
    print_tensor_table(model);
}

static std::string gguf_scalar_to_string(
        const struct gguf_context * ctx,
        int64_t key_id,
        enum gguf_type type) {
    char buffer[128];
    switch (type) {
        case GGUF_TYPE_UINT8:   std::snprintf(buffer, sizeof(buffer), "%" PRIu8,  gguf_get_val_u8 (ctx, key_id)); return buffer;
        case GGUF_TYPE_INT8:    std::snprintf(buffer, sizeof(buffer), "%" PRIi8,  gguf_get_val_i8 (ctx, key_id)); return buffer;
        case GGUF_TYPE_UINT16:  std::snprintf(buffer, sizeof(buffer), "%" PRIu16, gguf_get_val_u16(ctx, key_id)); return buffer;
        case GGUF_TYPE_INT16:   std::snprintf(buffer, sizeof(buffer), "%" PRIi16, gguf_get_val_i16(ctx, key_id)); return buffer;
        case GGUF_TYPE_UINT32:  std::snprintf(buffer, sizeof(buffer), "%" PRIu32, gguf_get_val_u32(ctx, key_id)); return buffer;
        case GGUF_TYPE_INT32:   std::snprintf(buffer, sizeof(buffer), "%" PRIi32, gguf_get_val_i32(ctx, key_id)); return buffer;
        case GGUF_TYPE_UINT64:  std::snprintf(buffer, sizeof(buffer), "%" PRIu64, gguf_get_val_u64(ctx, key_id)); return buffer;
        case GGUF_TYPE_INT64:   std::snprintf(buffer, sizeof(buffer), "%" PRIi64, gguf_get_val_i64(ctx, key_id)); return buffer;
        case GGUF_TYPE_FLOAT32: std::snprintf(buffer, sizeof(buffer), "%.9g",  gguf_get_val_f32(ctx, key_id)); return buffer;
        case GGUF_TYPE_FLOAT64: std::snprintf(buffer, sizeof(buffer), "%.17g", gguf_get_val_f64(ctx, key_id)); return buffer;
        case GGUF_TYPE_BOOL:    return gguf_get_val_bool(ctx, key_id) ? "true" : "false";
        case GGUF_TYPE_STRING:  return "\"" + escape_string(gguf_get_val_str(ctx, key_id)) + "\"";
        case GGUF_TYPE_ARRAY:
        case GGUF_TYPE_COUNT:   return "<unsupported>";
    }
    return "<invalid>";
}

template <typename T>
static void print_array_numeric(const void * raw, size_t count) {
    const T * values = static_cast<const T *>(raw);
    for (size_t i = 0; i < count; ++i) {
        if (i != 0) {
            std::printf(", ");
        }
        if (std::numeric_limits<T>::is_integer) {
            if (std::numeric_limits<T>::is_signed) {
                std::printf("%" PRIi64, static_cast<int64_t>(values[i]));
            } else {
                std::printf("%" PRIu64, static_cast<uint64_t>(values[i]));
            }
        } else {
            std::printf("%.9g", static_cast<double>(values[i]));
        }
    }
}

static void print_gguf_array(const struct gguf_context * ctx, int64_t key_id) {
    const enum gguf_type type = gguf_get_arr_type(ctx, key_id);
    const size_t n = gguf_get_arr_n(ctx, key_id);
    const size_t shown = std::min(n, VALUE_PREVIEW_COUNT);
    // String arrays are stored as std::string objects internally and must be
    // accessed with gguf_get_arr_str(); gguf_get_arr_data() asserts for them.
    const void * raw = type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx, key_id);

    std::printf("array<%s>[%zu] = [", gguf_type_name(type), n);
    switch (type) {
        case GGUF_TYPE_UINT8:   print_array_numeric<uint8_t> (raw, shown); break;
        case GGUF_TYPE_INT8:    print_array_numeric<int8_t>  (raw, shown); break;
        case GGUF_TYPE_UINT16:  print_array_numeric<uint16_t>(raw, shown); break;
        case GGUF_TYPE_INT16:   print_array_numeric<int16_t> (raw, shown); break;
        case GGUF_TYPE_UINT32:  print_array_numeric<uint32_t>(raw, shown); break;
        case GGUF_TYPE_INT32:   print_array_numeric<int32_t> (raw, shown); break;
        case GGUF_TYPE_UINT64:  print_array_numeric<uint64_t>(raw, shown); break;
        case GGUF_TYPE_INT64:   print_array_numeric<int64_t> (raw, shown); break;
        case GGUF_TYPE_FLOAT32: print_array_numeric<float>   (raw, shown); break;
        case GGUF_TYPE_FLOAT64: print_array_numeric<double>  (raw, shown); break;
        case GGUF_TYPE_BOOL: {
            const int8_t * values = static_cast<const int8_t *>(raw);
            for (size_t i = 0; i < shown; ++i) {
                std::printf("%s%s", i == 0 ? "" : ", ", values[i] ? "true" : "false");
            }
        } break;
        case GGUF_TYPE_STRING:
            for (size_t i = 0; i < shown; ++i) {
                const std::string value = gguf_get_arr_str(ctx, key_id, i);
                std::printf("%s\"%s\"", i == 0 ? "" : ", ", escape_string(value).c_str());
            }
            break;
        case GGUF_TYPE_ARRAY:
        case GGUF_TYPE_COUNT:
            std::printf("<unsupported>");
            break;
    }
    if (shown < n) {
        std::printf("%s... %zu more", shown == 0 ? "" : ", ", n - shown);
    }
    std::printf("]");
}

static struct gguf_context * parse_gguf(
        const std::string & path,
        model_info & model,
        struct ggml_context ** schema_ctx) {
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ schema_ctx,
    };
    struct gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    if (gguf == nullptr || *schema_ctx == nullptr) {
        std::fprintf(stderr, "error: gguf_init_from_file() failed for '%s'\n", path.c_str());
        if (*schema_ctx != nullptr) {
            ggml_free(*schema_ctx);
            *schema_ctx = nullptr;
        }
        if (gguf != nullptr) {
            gguf_free(gguf);
        }
        return nullptr;
    }

    model.data_offset = gguf_get_data_offset(gguf);
    const int64_t n_tensors = gguf_get_n_tensors(gguf);
    model.tensors.reserve(static_cast<size_t>(n_tensors));
    for (int64_t i = 0; i < n_tensors; ++i) {
        tensor_info info;
        info.name = gguf_get_tensor_name(gguf, i);
        info.type = gguf_get_tensor_type(gguf, i);
        const struct ggml_tensor * tensor = ggml_get_tensor(*schema_ctx, info.name.c_str());
        if (tensor == nullptr) {
            std::fprintf(stderr, "error: GGUF tensor '%s' is missing from its schema context\n", info.name.c_str());
            ggml_free(*schema_ctx);
            *schema_ctx = nullptr;
            gguf_free(gguf);
            return nullptr;
        }
        info.n_dims = ggml_n_dims(tensor);
        std::copy(tensor->ne, tensor->ne + GGML_MAX_DIMS, info.ne.begin());
        const uint64_t tensor_offset = gguf_get_tensor_offset(gguf, i);
        if (tensor_offset > std::numeric_limits<uint64_t>::max() - model.data_offset) {
            std::fprintf(stderr, "error: GGUF tensor '%s' file offset overflows uint64_t\n", info.name.c_str());
            ggml_free(*schema_ctx);
            *schema_ctx = nullptr;
            gguf_free(gguf);
            return nullptr;
        }
        info.file_offset = model.data_offset + tensor_offset;
        info.nbytes = gguf_get_tensor_size(gguf, i);
        if (info.file_offset > model.file_size || info.nbytes > model.file_size - info.file_offset) {
            std::fprintf(stderr, "error: GGUF tensor '%s' lies outside the file\n", info.name.c_str());
            ggml_free(*schema_ctx);
            *schema_ctx = nullptr;
            gguf_free(gguf);
            return nullptr;
        }
        model.tensors.push_back(std::move(info));
    }
    return gguf;
}

static void visualize_gguf_file(const model_info & model, const struct gguf_context * gguf) {
    std::printf("\n========== 1. GGUF FILE VISUALIZATION ==========\n\n");
    std::printf("  %s\n", model.path.c_str());
    std::printf("  [magic: GGUF]\n");
    std::printf("       |-- version     = %" PRIu32 "\n", gguf_get_version(gguf));
    std::printf("       |-- alignment   = %zu B\n", gguf_get_alignment(gguf));
    std::printf("       |-- kv pairs    = %" PRIi64 "\n", gguf_get_n_kv(gguf));
    std::printf("       `-- tensors     = %" PRIi64 "\n", gguf_get_n_tensors(gguf));
    std::printf("  [metadata and tensor directory]\n");
    std::printf("  [padding to 0x%012" PRIx64 "]\n", model.data_offset);
    std::printf("  [aligned tensor data]\n");
    std::printf("  [file end: 0x%012" PRIx64 ", %s]\n",
            model.file_size, format_bytes(model.file_size).c_str());

    std::printf("\n  key/value metadata\n");
    for (int64_t i = 0; i < gguf_get_n_kv(gguf); ++i) {
        const enum gguf_type type = gguf_get_kv_type(gguf, i);
        std::printf("  [%3" PRIi64 "] %-42s : ", i, gguf_get_key(gguf, i));
        if (type == GGUF_TYPE_ARRAY) {
            print_gguf_array(gguf, i);
        } else {
            std::printf("%s = %s", gguf_type_name(type), gguf_scalar_to_string(gguf, i, type).c_str());
        }
        std::printf("\n");
    }
    print_tensor_table(model);
}

static bool context_mem_size(const model_info & model, bool no_alloc, size_t & mem_size) {
    mem_size = 0;
    for (const tensor_info & info : model.tensors) {
        if (!add_size(mem_size, ggml_tensor_overhead())) {
            return false;
        }
        if (!no_alloc) {
            if (info.nbytes > std::numeric_limits<size_t>::max() - (GGML_MEM_ALIGN - 1)) {
                return false;
            }
            if (!add_size(mem_size, pad_to(info.nbytes, GGML_MEM_ALIGN))) {
                return false;
            }
        }
    }
    return true;
}

static struct ggml_context * build_context(const model_info & model, bool no_alloc) {
    size_t mem_size = 0;
    if (!context_mem_size(model, no_alloc, mem_size)) {
        std::fprintf(stderr, "error: context memory size overflow\n");
        return nullptr;
    }

    // This is the pair of ggml_init() calls the example is intended to compare.
    // In both cases mem_buffer is NULL, so ggml owns the allocated memory pool.
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ no_alloc,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "error: ggml_init(no_alloc=%s) failed\n", no_alloc ? "true" : "false");
        return nullptr;
    }

    std::ifstream input;
    if (!no_alloc) {
        input.open(model.path, std::ios::binary);
        if (!input) {
            std::fprintf(stderr, "error: cannot reopen '%s' for tensor data\n", model.path.c_str());
            ggml_free(ctx);
            return nullptr;
        }
    }

    for (const tensor_info & info : model.tensors) {
        struct ggml_tensor * tensor = ggml_new_tensor(ctx, info.type, info.n_dims, info.ne.data());
        if (tensor == nullptr) {
            std::fprintf(stderr, "error: cannot create tensor '%s'\n", info.name.c_str());
            ggml_free(ctx);
            return nullptr;
        }
        ggml_set_name(tensor, info.name.c_str());
        if (ggml_nbytes(tensor) != info.nbytes) {
            std::fprintf(stderr,
                    "error: tensor '%s' size changed while building context (%zu != %zu)\n",
                    info.name.c_str(), ggml_nbytes(tensor), info.nbytes);
            ggml_free(ctx);
            return nullptr;
        }

        if (!no_alloc) {
            if (info.nbytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
                std::fprintf(stderr, "error: tensor '%s' is too large for std::ifstream\n", info.name.c_str());
                ggml_free(ctx);
                return nullptr;
            }
            input.seekg(static_cast<std::streamoff>(info.file_offset), std::ios::beg);
            input.read(static_cast<char *>(tensor->data), static_cast<std::streamsize>(info.nbytes));
            if (!input) {
                std::fprintf(stderr, "error: cannot load data for tensor '%s'\n", info.name.c_str());
                ggml_free(ctx);
                return nullptr;
            }
        }
    }
    return ctx;
}

static void visualize_context(const struct ggml_context * ctx, bool requested_no_alloc) {
    const uintptr_t buffer_begin = reinterpret_cast<uintptr_t>(ggml_get_mem_buffer(ctx));
    const size_t mem_size = ggml_get_mem_size(ctx);
    const uintptr_t buffer_end = buffer_begin + mem_size;
    const size_t used = ggml_used_mem(ctx);

    std::printf("\n========== 2.%c GGML CONTEXT: no_alloc=%s ==========\n\n",
            requested_no_alloc ? 'A' : 'B', requested_no_alloc ? "true" : "false");
    std::printf("  ggml_init_params\n");
    std::printf("       |-- mem_size   = %s\n", format_bytes(mem_size).c_str());
    std::printf("       |-- mem_buffer = NULL (ggml allocated/owns the pool)\n");
    std::printf("       `-- no_alloc   = %s\n", requested_no_alloc ? "true" : "false");
    std::printf("  context\n");
    std::printf("       |-- actual no_alloc = %s\n", ggml_get_no_alloc(const_cast<struct ggml_context *>(ctx)) ? "true" : "false");
    std::printf("       |-- buffer range    = [%p, %p)\n",
            reinterpret_cast<void *>(buffer_begin), reinterpret_cast<void *>(buffer_end));
    std::printf("       |-- used memory     = %s\n", format_bytes(used).c_str());
    std::printf("       `-- free memory     = %s\n", format_bytes(mem_size - used).c_str());
    std::printf("\n  memory map (offsets are relative to context mem_buffer)\n");
    std::printf("  %-4s %-42s %-13s %-18s %-18s %s\n",
            "id", "tensor", "payload", "metadata", "data", "allocation");
    std::printf("  %s\n", std::string(116, '-').c_str());

    size_t id = 0;
    for (struct ggml_tensor * tensor = ggml_get_first_tensor(ctx);
         tensor != nullptr;
         tensor = ggml_get_next_tensor(ctx, tensor), ++id) {
        const uintptr_t metadata = reinterpret_cast<uintptr_t>(tensor);
        const uintptr_t data = reinterpret_cast<uintptr_t>(tensor->data);
        const bool data_in_pool = data >= buffer_begin && data < buffer_end;

        char metadata_text[32];
        char data_text[32];
        std::snprintf(metadata_text, sizeof(metadata_text), "+0x%012" PRIxPTR, metadata - buffer_begin);
        if (tensor->data == nullptr) {
            std::snprintf(data_text, sizeof(data_text), "NULL");
        } else if (data_in_pool) {
            std::snprintf(data_text, sizeof(data_text), "+0x%012" PRIxPTR, data - buffer_begin);
        } else {
            std::snprintf(data_text, sizeof(data_text), "%p", tensor->data);
        }

        std::printf("  %-4zu %-42.42s %-13zu %-18s %-18s %s\n",
                id, ggml_get_name(tensor), ggml_nbytes(tensor), metadata_text, data_text,
                tensor->data == nullptr ? "metadata only" : (data_in_pool ? "inside context" : "external"));
    }

    std::printf("\n  result: tensor metadata %s; tensor payload %s.\n",
            "is always inside the context pool",
            requested_no_alloc ? "is not allocated (data == NULL)" : "is allocated directly after each tensor metadata block");
}

static bool run_context_comparison(const model_info & model) {
    for (const bool no_alloc : { true, false }) {
        struct ggml_context * ctx = build_context(model, no_alloc);
        if (ctx == nullptr) {
            return false;
        }
        visualize_context(ctx, no_alloc);
        ggml_free(ctx);
    }
    return true;
}

static void print_usage(const char * program) {
    std::printf("usage: %s [model.gguf|legacy-gpt2.bin]\n", program);
    std::printf("default: %s\n", DEFAULT_MODEL_PATH);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc > 2 || (argc == 2 && (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0))) {
        print_usage(argv[0]);
        return argc > 2 ? 1 : 0;
    }

    model_info model;
    model.path = argc == 2 ? argv[1] : DEFAULT_MODEL_PATH;

    std::array<char, 4> magic {{ 0, 0, 0, 0 }};
    if (!inspect_magic(model.path, magic, model.file_size)) {
        return 1;
    }

    if (std::memcmp(magic.data(), GGUF_MAGIC, magic.size()) == 0) {
        struct ggml_context * schema_ctx = nullptr;
        struct gguf_context * gguf = parse_gguf(model.path, model, &schema_ctx);
        if (gguf == nullptr) {
            return 1;
        }
        visualize_gguf_file(model, gguf);
        ggml_free(schema_ctx);
        gguf_free(gguf);
    } else {
        uint32_t magic_value = 0;
        std::memcpy(&magic_value, magic.data(), sizeof(magic_value));
        if (magic_value != GGML_FILE_MAGIC) {
            std::fprintf(stderr,
                    "error: unsupported magic '%c%c%c%c'; expected GGUF or legacy GGML\n",
                    magic[0], magic[1], magic[2], magic[3]);
            return 1;
        }

        legacy_gpt2_info legacy;
        if (!parse_legacy_ggml(model.path, model, legacy)) {
            return 1;
        }
        visualize_legacy_file(model, legacy);
    }

    return run_context_comparison(model) ? 0 : 1;
}

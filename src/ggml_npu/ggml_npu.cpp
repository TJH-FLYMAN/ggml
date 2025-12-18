#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

#include "ggml-npu.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-npu-quants.h"
#include "ggml-threading.h"
#include "amx/amx.h"
#include "ggml.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#include <unistd.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

// ggml-backend interface



// npu backend - backend (stream)

struct ggml_backend_npu_context {
    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};

static const char * ggml_backend_npu_get_name(ggml_backend_t backend) {
    return "NPU";

    GGML_UNUSED(backend);
}

static void ggml_backend_npu_free(ggml_backend_t backend) {
    struct ggml_backend_npu_context * npu_ctx = (struct ggml_backend_npu_context *)backend->context;
    delete[] npu_ctx->work_data;
    delete npu_ctx;
    delete backend;
}

struct ggml_backend_plan_npu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_backend_graph_plan_t ggml_backend_npu_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    // 未实现
    return npu_plan;
}

static void ggml_backend_npu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_npu * npu_plan = (struct ggml_backend_plan_npu *)plan;

    delete[] npu_plan->cplan.work_data;
    delete npu_plan;

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_npu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
// 未实现
}

static enum ggml_status ggml_backend_npu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
// 未实现
}

static const struct ggml_backend_i ggml_backend_npu_i = {
    /* .get_name                = */ ggml_backend_npu_get_name,
    /* .free                    = */ ggml_backend_npu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_npu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_npu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_npu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_npu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

// npu标识符 随机生成的
static ggml_guid_t ggml_backend_npu_guid(void) {
    static ggml_guid guid = {0x52, 0x3c, 0xa1, 0xf7, 0x25, 0x80, 0x43, 0xa8,0xb3, 0x9e, 0xe3, 0x65, 0xce, 0x45, 0xcf, 0xbb};
    return &guid;
}

ggml_backend_t ggml_backend_npu_init(void) {
    // initialize npu backend now to avoid slowing the first graph computation
    ggml_npu_init();

    struct ggml_backend_npu_context * ctx = new ggml_backend_npu_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;

    ggml_backend_t npu_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_npu_guid(),
        /* .interface = */ ggml_backend_npu_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_npu_reg(), 0),
        /* .context   = */ ctx,
    };

    if (npu_backend == NULL) {
        delete ctx;
        return NULL;
    }

    return npu_backend;
}

void ggml_npu_init(void) {
    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start(); // mutex.lock()

    static bool is_first_call = true;

    if (is_first_call) {
#if defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

        is_first_call = false;
    }

    ggml_critical_section_end(); // mutex.lock()
} 

bool ggml_backend_is_npu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_npu_guid());
}

static bool ggml_backend_npu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft) || ggml_backend_npu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_npu_device_i = {
    /* .get_name             = */ ggml_backend_npu_device_get_name,
    /* .get_description      = */ ggml_backend_npu_device_get_description,
    /* .get_memory           = */ ggml_backend_npu_device_get_memory,
    /* .get_type             = */ ggml_backend_npu_device_get_type,
    /* .get_props            = */ ggml_backend_npu_device_get_props,
    /* .init_backend         = */ ggml_backend_npu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_npu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_npu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_npu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_npu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

struct ggml_backend_npu_device_context {
    std::string description = "NPU";

    ggml_backend_npu_device_context() {
#if defined(__linux__)
#if USE_AI985
    if (access("/dev/vipcore", F_OK) == 0) {
        description += "AI985 vipcore"
    } else {
        description  = "";
    }
#endif
#endif
    }
};

static const char * ggml_backend_npu_device_get_name(ggml_backend_dev_t dev) {
    return "npu";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_npu_device_get_description(ggml_backend_dev_t dev) {
    struct ggml_backend_npu_device_context * ctx = (struct ggml_backend_npu_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_npu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_npu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_NPU;

    GGML_UNUSED(dev);
}

// npu backend - backend (reg)

static const char * ggml_backend_npu_reg_get_name(ggml_backend_reg_t reg) {
    return "NPU";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_npu_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}
static ggml_backend_dev_t ggml_backend_npu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_npu_device_context ctx;
    static ggml_backend_device ggml_backend_npu_device = {
        /* .iface   = */ ggml_backend_npu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    return &ggml_backend_npu_device;
}

static void * ggml_backend_npu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "r853") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "r853_dy") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "r853_seg") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_npu_reg_i = {
    /* .get_name         = */ ggml_backend_npu_reg_get_name,
    /* .get_device_count = */ ggml_backend_npu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_npu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_npu_get_proc_address,
};


ggml_backend_reg_t ggml_backend_npu_reg(void) {
    // init npu feature detection
    ggml_npu_init();

    static struct ggml_backend_reg ggml_backend_npu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_npu_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_npu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_npu_reg)
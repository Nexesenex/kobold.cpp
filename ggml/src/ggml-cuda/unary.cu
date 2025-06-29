#include "unary.cuh"

template <class T>
static __global__ void op_abs(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = fabsf(x[i]);
}

template <class T>
static __global__ void op_sgn(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)(x[i] > (T)0.f ? 1.f : ((x[i] < (T)0.f ? -1.f : 0.f)));
}

template <class T>
static __global__ void op_neg(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = -x[i];
}

template <class T>
static __global__ void op_step(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = x[i] > (T)0.0f;
}

template <class T>
static __global__ void op_gelu(const T * x, T * dst, const int k) {
    const T GELU_COEF_A    = 0.044715f;
    const T SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    T xi = x[i];
    dst[i] = (T)0.5f*xi*((T)1.0f + (T)tanhf(SQRT_2_OVER_PI*xi*((T)1.0f + GELU_COEF_A*xi*xi)));
}

template <class T>
static __global__ void gelu_erf_f32(const T * x, T * dst, const int k) {
    const T SQRT_2_INV = 0.70710678118654752440084436210484f;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = (T) 0.5f * x[i] * ((T)1.0f + (T)erff(SQRT_2_INV * x[i]));
}

// static __global__ void gelu_quick_f32(const float * x, float * dst, int k) {
    // const float GELU_QUICK_COEF = -1.702f;

template <class T>
static __global__ void op_gelu_quick(const T * x, T * dst, int k) {
    const T GELU_QUICK_COEF = -1.702f;
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = x[i] * ((T)1.0f / ((T)1.0f + (T)expf(GELU_QUICK_COEF * x[i])));
}

template <class T>
static __global__ void op_silu(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] / ((T)1.0f + (T)expf(-x[i]));
}

template <class T>
static __global__ void op_silu_back(
        const T * grad, const T * xf, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const T xfi = xf[i];
    const T s = (T)1.0f / ((T)1.0f + (T)expf(-xfi));
    dst[i] = grad[i] * s * ((T)1.0f + xfi * ((T)1.0f - s));
}

template <class T>
static __global__ void fused_mul_silu_f32(const T * x, const T * y, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * y[i] / ((T)1.0f + (T)expf(-x[i]));
}

static __global__ void multi_add_f32(int nused, int64_t ne0, int64_t ne1, int64_t nb1, int64_t nb01, const char * src0, char * dst) {

    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;
    int64_t k = ne0*ne1;
    if (i >= k) {
        return;
    }
    int i1 = i / ne0;
    int i0 = i % ne0;
    float * result = (float *)(dst + i1*nb1);
    const float * s = (const float *)(src0 + i1*nb01) + i0;
    if (nused == 1) {
        result[i0] = s[0];
    } else {
        float sum = s[0] + s[ne0];
        for (int j = 2; j < nused; ++j) sum += s[j*ne0];
        result[i0] = sum;
    }
}

// template <class T>
// static __global__ void multi_add_f32(int nused, int64_t ne0, int64_t ne1, int64_t nb1, int64_t nb01, const char * src0, T * dst) {
    // const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;
    // int64_t k = ne0*ne1;
    // if (i >= k) {
        // return;
    // }
    // int i1 = i / ne0;
    // int i0 = i % ne0;
    // T * result = (T *)(dst + i1*nb1);
    // const float * s = (const float *)(src0 + i1*nb01) + i0;
    // if (nused == 1) {
        // result[i0] = s[0];
    // } else {
        // T sum = s[0] + s[ne0];
        // for (int j = 2; j < nused; ++j) sum += s[j*ne0];
        // result[i0] = sum;
    // }
// }

template <class T>
static __global__ void fused_mul_relu_f32(const T * x, const T * y, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fmaxf(x[i], 0) * y[i];
}

template <class T>
static __global__ void fused_mul_gelu_f32(const T * x, const T * y, T * dst, const int k) {
    constexpr T GELU_COEF_A    = 0.044715f;
    constexpr T SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    T xi = x[i];
    dst[i] = (T)0.5f*xi*y[i]*((T)1.0f + (T)tanhf(SQRT_2_OVER_PI*xi*((T)1.0f + (T)GELU_COEF_A*xi*xi)));
}

// static __global__ void tanh_f32(const float * x, float * dst, int k) {

template <class T>
static __global__ void op_tanh(const T * x, T * dst, int k) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = tanhf(x[i]);
}

template <class T>
static __global__ void op_relu(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fmaxf(x[i], 0);
}

template <class T>
static __global__ void op_sigmoid(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = (T)1.0f / ((T)1.0f + (T)expf(-x[i]));
}

template <class T>
static __global__ void op_hardsigmoid(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + (T)3.0f) / (T)6.0f));
}

template <class T>
static __global__ void op_hardswish(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * (T)fminf(1.0f, fmaxf(0.0f, (x[i] + (T)3.0f) / (T)6.0f));
}

template <class T>
static __global__ void op_exp(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = expf(x[i]);
}

template <class T>
static __global__ void op_leaky_relu(const T * x, T * dst, const int k, const float negative_slope) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = (T)fmaxf(x[i], 0) + (T)fminf(x[i], 0.0f) * (T)negative_slope;
}

template <class T>
static __global__ void op_sqr(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * x[i];
}

template <class T>
static __global__ void op_sqrt(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = sqrtf(x[i]);
}

template <class T>
static __global__ void op_sin(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = sinf(x[i]);
}

template <class T>
static __global__ void op_cos(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = cosf(x[i]);
}

template <class T>
static __global__ void op_log(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = logf(x[i]);
}

template <class T>
static void abs_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    op_abs<<<num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void sgn_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    op_sgn<<<num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void neg_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    op_neg<<<num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void step_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_STEP_BLOCK_SIZE - 1) / CUDA_STEP_BLOCK_SIZE;
    op_step<<<num_blocks, CUDA_STEP_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void gelu_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    op_gelu<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void gelu_erf_f32_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    gelu_erf_f32<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

// static void gelu_quick_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {

template <class T>
static void gelu_quick_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    op_gelu_quick<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void silu_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    op_silu<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void silu_back_cuda(const T * grad, const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    op_silu_back<<<num_blocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, stream>>>(grad, x, dst, k);
}

template <class T>
static void fused_mul_silu_f32_cuda(const T * x, const T * y, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    fused_mul_silu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, y, dst, k);
}

template <class T>
static void fused_mul_relu_f32_cuda(const T * x, const T * y, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    fused_mul_relu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, y, dst, k);
}

template <class T>
static void fused_mul_gelu_f32_cuda(const T * x, const T * y, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    fused_mul_gelu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, y, dst, k);
}

// static void tanh_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {

template <class T>
static void tanh_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_TANH_BLOCK_SIZE - 1) / CUDA_TANH_BLOCK_SIZE;
    op_tanh<<<num_blocks, CUDA_TANH_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void relu_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    op_relu<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void sigmoid_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SIGMOID_BLOCK_SIZE - 1) / CUDA_SIGMOID_BLOCK_SIZE;
    op_sigmoid<<<num_blocks, CUDA_SIGMOID_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void hardsigmoid_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_HARDSIGMOID_BLOCK_SIZE - 1) / CUDA_HARDSIGMOID_BLOCK_SIZE;
    op_hardsigmoid<<<num_blocks, CUDA_HARDSIGMOID_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void hardswish_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_HARDSWISH_BLOCK_SIZE - 1) / CUDA_HARDSWISH_BLOCK_SIZE;
    op_hardswish<<<num_blocks, CUDA_HARDSWISH_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void exp_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_EXP_BLOCK_SIZE - 1) / CUDA_EXP_BLOCK_SIZE;
    op_exp<<<num_blocks, CUDA_EXP_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void leaky_relu_cuda(const T * x, T * dst, const int k, const float negative_slope, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    op_leaky_relu<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, negative_slope);
}

template <class T>
static void sqr_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SQR_BLOCK_SIZE - 1) / CUDA_SQR_BLOCK_SIZE;
    op_sqr<<<num_blocks, CUDA_SQR_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void sqrt_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SQRT_BLOCK_SIZE - 1) / CUDA_SQRT_BLOCK_SIZE;
    op_sqrt<<<num_blocks, CUDA_SQRT_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void sin_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SIN_BLOCK_SIZE - 1) / CUDA_SIN_BLOCK_SIZE;
    op_sin<<<num_blocks, CUDA_SIN_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void cos_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_COS_BLOCK_SIZE - 1) / CUDA_COS_BLOCK_SIZE;
    op_cos<<<num_blocks, CUDA_COS_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void log_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_COS_BLOCK_SIZE - 1) / CUDA_COS_BLOCK_SIZE;
    op_log<<<num_blocks, CUDA_COS_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

void ggml_cuda_op_abs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        abs_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        abs_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_sgn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        sgn_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        sgn_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        neg_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        neg_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        step_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        step_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void multi_add_f32_cuda(int nused, int64_t ne0, int64_t ne1, int64_t nb1, int64_t nb01, const char * src0, char * dst, cudaStream_t stream) {
    int64_t k = ne0 * ne1;
    const int num_blocks = (k + CUDA_MULTI_ADD_BLOCK_SIZE - 1) / CUDA_MULTI_ADD_BLOCK_SIZE;
    multi_add_f32<<<num_blocks, CUDA_MULTI_ADD_BLOCK_SIZE, 0, stream>>>(nused, ne0, ne1, nb1, nb01, src0, dst);
}

void ggml_cuda_op_multi_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[2] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    int nused = dst->op_params[0];
    GGML_ASSERT(nused >= 1);
    const char * src0 = (const char *)dst->src[0]->data;
    cudaStream_t stream = ctx.stream();
    multi_add_f32_cuda(nused, dst->ne[0], dst->ne[1], dst->nb[1], dst->src[0]->nb[1], (char *)src0, (char *)dst->data, stream);
}

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        gelu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        gelu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        silu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        silu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // input from forward pass
    const ggml_tensor * src1 = dst->src[1]; // grads of forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        silu_back_cuda((const half *)src0_d, (const half *)src1_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        silu_back_cuda((const float*)src0_d, (const float*)src1_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_fused_mul_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));

    cudaStream_t stream = ctx.stream();
    ggml_unary_op op = (ggml_unary_op)dst->op_params[0];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;

    switch (op) {
        case GGML_UNARY_OP_SILU: fused_mul_silu_f32_cuda(src0_d, src1_d, dst_d, ggml_nelements(dst), stream); break;
        case GGML_UNARY_OP_RELU: fused_mul_relu_f32_cuda(src0_d, src1_d, dst_d, ggml_nelements(dst), stream); break;
        case GGML_UNARY_OP_GELU: fused_mul_gelu_f32_cuda(src0_d, src1_d, dst_d, ggml_nelements(dst), stream); break;
        default: GGML_ASSERT(false);
    }
}

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_erf_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        gelu_quick_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        gelu_quick_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        tanh_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        tanh_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        relu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        relu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        sigmoid_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        sigmoid_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        hardsigmoid_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        hardsigmoid_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        hardswish_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        hardswish_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        exp_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        exp_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        leaky_relu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), negative_slope, stream);
    } else {
        leaky_relu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), negative_slope, stream);
    }
}

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        sqr_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        sqr_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        sqrt_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        sqrt_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        sin_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        sin_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        cos_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        cos_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_log(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        log_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        log_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

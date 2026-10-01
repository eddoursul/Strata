// src/kernels/cuda/kv_q4.cu - see include/strata/kernels/kv_q4.hpp.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "fwht.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_q4: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void validate(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256 || s.n_head_kv <= 0 || s.page_size <= 0) {
        std::fprintf(stderr, "kv_q4: %s: head_dim must be 256 (the transform's size)\n", what);
        std::exit(1);
    }
}

// One warp per (token, KV head, side): the head's 256 values (lane l: values l + 32 i) rotated, then block b - register
// b of every lane - quantized: its value of largest magnitude (on a tie the larger value, so every lane agrees) over
// -8 is the scale.  STEP: the position from the token's step record, else pos0 + token.
template <bool STEP>
__global__ void __launch_bounds__(32) kv_append_q4_kernel(uint8_t* __restrict__ k_q4, uint16_t* __restrict__ k_q4s,
                                                          uint8_t* __restrict__ v_q4, uint16_t* __restrict__ v_q4s,
                                                          const int32_t* __restrict__ table,
                                                          const int32_t* __restrict__ step, int step_stride,
                                                          int64_t pos0, const float* __restrict__ K,
                                                          const float* __restrict__ V, int64_t ld, int kv_heads,
                                                          int page_size) {
    const int t = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    const bool is_v = blockIdx.z + (k_q4 == nullptr ? 1 : 0) == 1;
    const float* x = (is_v ? V : K) + (size_t) t * ld + (size_t) h * 256;
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = x[32 * i + lane];
    fwht256_warp(v, lane);
    const long long pos = STEP ? (long long) __ldg(step + (size_t) t * step_stride + kStepPos) : (long long) (pos0 + t);
    const long long row = ((long long) table[pos / page_size] * kv_heads + h) * page_size + pos % page_size;
    uint8_t* codes = (is_v ? v_q4 : k_q4) + row * 128;
    uint16_t* scales = (is_v ? v_q4s : k_q4s) + row * 8;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        float a = fabsf(v[b]), m = v[b];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float a2 = __shfl_xor_sync(0xffffffffu, a, o), m2 = __shfl_xor_sync(0xffffffffu, m, o);
            if (a2 > a || (a2 == a && m2 > m)) { a = a2; m = m2; }
        }
        const uint16_t db = f16_from_f32(m / -8.0f);
        const float d = f32_from_f16(db);   // quantized against the STORED scale
        int q = 8;
        if (d != 0.0f) q = min(15, max(0, __float2int_rn(v[b] / d) + 8));
        const int hi = __shfl_down_sync(0xffffffffu, q, 16);
        if (lane < 16) codes[16 * b + lane] = (uint8_t) (q | (hi << 4));
        if (lane == 0) scales[b] = db;
    }
}

__global__ void __launch_bounds__(128) fwht256_rows_kernel(float* __restrict__ x, int64_t n_rows) {
    const int64_t r = (int64_t) blockIdx.x * 4 + (threadIdx.x >> 5);
    if (r >= n_rows) return;
    const int lane = threadIdx.x & 31;
    float* row = x + r * 256;
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = row[32 * i + lane];
    fwht256_warp(v, lane);
#pragma unroll
    for (int i = 0; i < 8; ++i) row[32 * i + lane] = v[i];
}

}  // namespace

void kv_append_q4_steps(uint8_t* k_q4, uint16_t* k_q4s, uint8_t* v_q4, uint16_t* v_q4s, const int32_t* page_table,
                        const int32_t* step, int step_stride, const float* kcur, const float* vcur, int n_tok,
                        const QsaShapes& s, void* stream) {
    validate(s, "kv_append_q4_steps");
    const unsigned sides = (k_q4 != nullptr ? 1u : 0u) + (v_q4 != nullptr ? 1u : 0u);
    if (n_tok < 1 || sides == 0) return;
    const dim3 grid((unsigned) n_tok, (unsigned) s.n_head_kv, sides);
    kv_append_q4_kernel<true><<<grid, 32, 0, (cudaStream_t) stream>>>(
        k_q4, k_q4s, v_q4, v_q4s, page_table, step, step_stride, 0, kcur, vcur, s.n_head_kv * s.head_dim,
        (int) s.n_head_kv, (int) s.page_size);
    check("kv_append_q4_steps launch");
}

void kv_append_q4_rows(uint8_t* k_q4, uint16_t* k_q4s, uint8_t* v_q4, uint16_t* v_q4s, const int32_t* page_table,
                       int64_t pos0, int64_t T, const float* K, const float* V, int64_t ld, const QsaShapes& s,
                       void* stream) {
    validate(s, "kv_append_q4_rows");
    const unsigned sides = (k_q4 != nullptr ? 1u : 0u) + (v_q4 != nullptr ? 1u : 0u);
    if (T < 1 || sides == 0) return;
    for (int64_t t0 = 0; t0 < T; t0 += 65535) {
        const int64_t n = T - t0 < 65535 ? T - t0 : 65535;
        const dim3 grid((unsigned) n, (unsigned) s.n_head_kv, sides);
        kv_append_q4_kernel<false><<<grid, 32, 0, (cudaStream_t) stream>>>(
            k_q4, k_q4s, v_q4, v_q4s, page_table, nullptr, 0, pos0 + t0, K + t0 * ld, V + t0 * ld, ld,
            (int) s.n_head_kv, (int) s.page_size);
    }
    check("kv_append_q4_rows launch");
}

void fwht256_rows(float* x, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    fwht256_rows_kernel<<<(unsigned) ((n_rows + 3) / 4), 128, 0, (cudaStream_t) stream>>>(x, n_rows);
    check("fwht256_rows launch");
}

}  // namespace strata::kernels

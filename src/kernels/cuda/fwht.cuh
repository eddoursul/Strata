// src/kernels/cuda/fwht.cuh - the orthonormal Walsh-Hadamard transform of a 256-value row in one warp: lane l holds
// values l + 32 i (i = 0..7) in v[i].  It is its own inverse; the 4-bit KV stores K and V rotated by it (kv_q4.hpp).
// The arithmetic of upstream's fwht256_kernel: the 1/16 scale first, then the butterflies over the lane bits and over
// the register bits.
#pragma once

__device__ __forceinline__ void fwht256_warp(float (&v)[8], int lane) {
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] *= 0.0625f;
#pragma unroll
    for (int h = 1; h < 32; h <<= 1)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, v[i], h);
            v[i] = (lane & h) ? o - v[i] : v[i] + o;
        }
#pragma unroll
    for (int h = 1; h < 8; h <<= 1)
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if ((i & h) == 0) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
}

// include/strata/kernels/kv_q4.hpp - 4-bit KV storage for the QSA layers: `--kv q4_0` (K and V) and `--kv k8v4` (V,
// with K in INT8, kv_q8.hpp).
//
// Upstream's formats (PR #21, code-martin; PR #120): a head's 256 values are rotated by the orthonormal Walsh-Hadamard
// transform H (fwht256_warp, src/kernels/cuda/fwht.cuh), which spreads its outlier channels over all of them, and then
// stored as 4-bit codes in blocks of 32 values with one FP16 scale each, ggml's q4_0 rounding against the stored scale:
//     d = fp16(the value of largest magnitude / -8),  code = clamp(rint(x / d) + 8, 0, 15),  x' = (code - 8) d
// <Hq, Hk> = <q, k>: the attention rotates its queries when K is rotated, and its output, a mix of rotated values,
// back (H is its own inverse), so the attention's callers see neither rotation.
//
// The codes and the scales are separate arrays, as in the INT8 pools, so the attention fetches both in 16-byte copies:
// codes [page][kv_head][page_size][head_dim / 2] (block b's byte j: value 32b + j in the low nibble, 32b + j + 16 in
// the high one), scales [page][kv_head][page_size][head_dim / 32].  144 bytes per cell and head, against 264 in INT8.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

inline constexpr int KV_Q4_GROUP = 32;

/// Bytes per cell (one token, one layer) of one 4-bit side, K or V: codes and scales of every KV head.
inline uint64_t kv_q4_bytes_per_side(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * ((uint64_t) s.head_dim / 2 + (uint64_t) (s.head_dim / KV_Q4_GROUP) * 2);
}

/// Appends n_tok cells (graph-capturable): token t's step record at step + t * step_stride (its position at
/// kStepPos), its K and V rows at kcur/vcur + t * n_head_kv * head_dim.  K is rotated and stored when k_q4 is
/// given, V when v_q4 is.
void kv_append_q4_steps(uint8_t* k_q4, uint16_t* k_q4s, uint8_t* v_q4, uint16_t* v_q4s, const int32_t* page_table,
                        const int32_t* step, int step_stride, const float* kcur, const float* vcur, int n_tok,
                        const QsaShapes& s, void* stream);

/// The prompt path's form: T cells from position pos0, token t's K and V rows at K/V + t * ld.
void kv_append_q4_rows(uint8_t* k_q4, uint16_t* k_q4s, uint8_t* v_q4, uint16_t* v_q4s, const int32_t* page_table,
                       int64_t pos0, int64_t T, const float* K, const float* V, int64_t ld, const QsaShapes& s,
                       void* stream);

/// H applied to n_rows rows of 256 floats in place (fwht256_warp: the tests' and the tools' form).
void fwht256_rows(float* x, int64_t n_rows, void* stream);

}  // namespace strata::kernels

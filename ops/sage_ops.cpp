// The attention kernel of SageAttention 2 on sm89 (INT8 Q K, FP8 P V, FP16 accumulation, the V scale fused;
// reference/SageAttention, Apache 2.0) as the operator live::sage_sm89, so that AOTInductor calls it from C++. The
// quantization of Q, K and V before it are Triton kernels and torch ops of the exported graph (kernels/sage_kv.py).
// live::sage_sm89_sparse is the same kernel over a list of key blocks per query block (ops/sage_attention_sm89.cuh).
#include "sage_sm89.h"

#include <torch/library.h>

namespace {

// q, k [B, tokens, heads, 128] INT8 and v [B, 128, heads, padded tokens] FP8 in Sage's layouts, with their scales;
// the output [B, q tokens, heads, 128] bfloat16
at::Tensor sage_sm89(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& q_scale,
                     const at::Tensor& k_scale, const at::Tensor& v_scale, double sm_scale) {
    auto out = at::empty(q.sizes(), q.options().dtype(at::kBFloat16));
    sage_sm89_dense(q, k, v, out, q_scale, k_scale, v_scale, static_cast<float>(sm_scale));
    return out;
}

// lut [B, heads, query blocks, key blocks] int32: the key blocks of each query block ascending, the last block of
// the keys among them; valid [B, heads, query blocks] int32: how many of them
at::Tensor sage_sm89_sparse(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& q_scale,
                            const at::Tensor& k_scale, const at::Tensor& v_scale, const at::Tensor& lut,
                            const at::Tensor& valid, double sm_scale) {
    auto out = at::empty(q.sizes(), q.options().dtype(at::kBFloat16));
    sage_sm89_blocks(q, k, v, out, q_scale, k_scale, v_scale, lut, valid, static_cast<float>(sm_scale));
    return out;
}

at::Tensor sage_sm89_sparse_meta(const at::Tensor& q, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                 const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, double) {
    return at::empty(q.sizes(), q.options().dtype(at::kBFloat16));
}

at::Tensor sage_sm89_meta(const at::Tensor& q, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                          const at::Tensor&, const at::Tensor&, double) {
    return at::empty(q.sizes(), q.options().dtype(at::kBFloat16));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(live, m) {
    m.def("sage_sm89(Tensor q, Tensor k, Tensor v, Tensor q_scale, Tensor k_scale, Tensor v_scale, float sm_scale) "
          "-> Tensor");
    m.def("sage_sm89_sparse(Tensor q, Tensor k, Tensor v, Tensor q_scale, Tensor k_scale, Tensor v_scale, Tensor lut, "
          "Tensor valid, float sm_scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(live, CUDA, m) {
    m.impl("sage_sm89", &sage_sm89);
    m.impl("sage_sm89_sparse", &sage_sm89_sparse);
}

TORCH_LIBRARY_IMPL(live, Meta, m) {
    m.impl("sage_sm89", &sage_sm89_meta);
    m.impl("sage_sm89_sparse", &sage_sm89_sparse_meta);
}

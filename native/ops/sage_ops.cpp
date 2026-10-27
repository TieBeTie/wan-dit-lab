// The attention kernel of SageAttention 2 on sm89 (INT8 Q K, FP8 P V, FP16 accumulation, the V scale fused;
// reference/SageAttention, Apache 2.0) as the operator live::sage_sm89, so that AOTInductor calls it from C++. The
// quantization of Q, K and V before it are Triton kernels and torch ops of the exported graph (kernels/sage_kv.py).
#include "attn_cuda_sm89.h"

#include <torch/library.h>

namespace {

// q, k [B, tokens, heads, 128] INT8 and v [B, 128, heads, padded tokens] FP8 in Sage's layouts, with their scales;
// the output [B, q tokens, heads, 128] bfloat16
at::Tensor sage_sm89(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& q_scale,
                     const at::Tensor& k_scale, const at::Tensor& v_scale, double sm_scale) {
    auto out = at::empty(q.sizes(), q.options().dtype(at::kBFloat16));
    // tensor layout NHD (0), not causal, per-thread granularity of the scales (3), no log-sum-exp
    qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(q, k, v, out, q_scale, k_scale, v_scale, 0, 0, 3,
                                                       static_cast<float>(sm_scale), 0);
    return out;
}

at::Tensor sage_sm89_meta(const at::Tensor& q, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                          const at::Tensor&, const at::Tensor&, double) {
    return at::empty(q.sizes(), q.options().dtype(at::kBFloat16));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(live, m) {
    m.def("sage_sm89(Tensor q, Tensor k, Tensor v, Tensor q_scale, Tensor k_scale, Tensor v_scale, float sm_scale) "
          "-> Tensor");
}

TORCH_LIBRARY_IMPL(live, CUDA, m) { m.impl("sage_sm89", &sage_sm89); }

TORCH_LIBRARY_IMPL(live, Meta, m) { m.impl("sage_sm89", &sage_sm89_meta); }

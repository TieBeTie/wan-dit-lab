// The one instance of SageAttention 2's sm89 kernel that Live runs (upstream's
// qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf with tensor layout NHD, not causal, per-thread scales, no
// log-sum-exp, bfloat16 out), dense or over a list of key blocks per query block (ops/sage_attention_sm89.cuh).
// SPARSE is a template argument: the dense operator is compiled without a line of the block lists.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "sage_attention_sm89.cuh"
#include "sage_sm89.h"

namespace {

template <bool SPARSE>
void launch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out, const at::Tensor& q_scale,
            const at::Tensor& k_scale, const at::Tensor& v_scale, float sm_scale, const int32_t* lut,
            const int32_t* valid) {
    constexpr int CTA_Q = 128, CTA_K = 64, WARP_Q = 32, WARP_K = 64, HEAD_DIM = 128;
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && q_scale.is_cuda() && k_scale.is_cuda() && v_scale.is_cuda());
    TORCH_CHECK(q.scalar_type() == at::kChar && k.scalar_type() == at::kChar, "q, k INT8");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && q.size(3) == HEAD_DIM && k.size(3) == HEAD_DIM);
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.is_contiguous() && q_scale.is_contiguous() &&
                k_scale.is_contiguous() && v_scale.is_contiguous());
    const int batch = q.size(0), qo_len = q.size(1), kv_len = k.size(1), heads = q.size(2), kv_heads = k.size(2);
    TORCH_CHECK(heads % kv_heads == 0 && v.size(1) == HEAD_DIM && v.size(2) == kv_heads &&
                v.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K, "v [B, 128, heads, padded tokens]");
    TORCH_CHECK(q_scale.sizes() == at::IntArrayRef({batch, heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8}));
    TORCH_CHECK(k_scale.sizes() == at::IntArrayRef({batch, kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4}));
    TORCH_CHECK(v_scale.sizes() == at::IntArrayRef({batch, kv_heads, HEAD_DIM}));

    size_t smem = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + 2 * CTA_K * HEAD_DIM * sizeof(int8_t),
                           CTA_Q * HEAD_DIM * sizeof(half));
    auto kernel = qk_int_sv_f8_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8,
                                           QuantGranularity::kPerThread, QuantGranularity::kPerThread, float, true,
                                           nv_bfloat16, ComputeUnit::kCudaCore, MaskMode::kNone, false, true, false,
                                           true, SPARSE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid(div_ceil(qo_len, CTA_Q), heads, batch), block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
    kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<int8_t>(), k.data_ptr<int8_t>(), reinterpret_cast<int8_t*>(v.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(out.data_ptr()), nullptr, q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale.data_ptr<float>(), nullptr, qo_len, kv_len, heads / kv_heads, q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(2), v.stride(1), out.stride(0), out.stride(1),
        out.stride(2), sm_scale, lut, valid);
}

}  // namespace

void sage_sm89_dense(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
                     const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale, float sm_scale) {
    launch<false>(q, k, v, out, q_scale, k_scale, v_scale, sm_scale, nullptr, nullptr);
}

void sage_sm89_blocks(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
                      const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale,
                      const at::Tensor& lut, const at::Tensor& valid, float sm_scale) {
    const int64_t batch = q.size(0), heads = q.size(2), blocks_q = div_ceil(q.size(1), 128),
                  blocks_k = div_ceil(k.size(1), 64);
    TORCH_CHECK(lut.is_cuda() && valid.is_cuda() && lut.scalar_type() == at::kInt && valid.scalar_type() == at::kInt &&
                lut.is_contiguous() && valid.is_contiguous());
    TORCH_CHECK(lut.sizes() == at::IntArrayRef({batch, heads, blocks_q, blocks_k}),
                "lut [B, heads, query blocks, key blocks]");
    TORCH_CHECK(valid.sizes() == at::IntArrayRef({batch, heads, blocks_q}), "valid [B, heads, query blocks]");
    launch<true>(q, k, v, out, q_scale, k_scale, v_scale, sm_scale, lut.data_ptr<int32_t>(), valid.data_ptr<int32_t>());
}

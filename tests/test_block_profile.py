"""block_profile: the kernels of a compiled block package grouped into the parts of a block."""

import pytest

from wan_dit_lab.harness import block_profile as bp

# kernel names as torch.profiler reports them for artifacts/live/608x1120-ordered-win24/block_2.pt2 (2026-11-12)
KERNELS = {
    "_Z24qk_int_sv_f8_attn_kernelILj128ELj64ELj32ELj64ELj128EL8DataType1E": "self-attention: Sage kernel",
    "_sum_max": "self-attention: Sage preparation",
    "_quant_k": "self-attention: Sage preparation",
    "_quant_v": "self-attention: Sage preparation",
    "triton_poi_fused__to_copy_add_clamp_clone_constant_pad_nd_div_ge_permute_sage_sm89_scalar_tensor_slice": "self-attention: Sage preparation",
    "triton_red_fused__to_copy_abs_amax_constant_pad_nd_stack_view_7": "self-attention: Sage preparation",
    "_rope_apply_kernel": "self-attention: RoPE",
    "_store_kernel": "KV store",
    "_ZN8nunchaku7kernels13invoke_kernelINS0_9GEMM_W4A4INS0_15GEMMConfig_W4A4ILb1ELb0EEEE16gemm_w4a4_kernel": "linear W4A4: GEMM",
    "_ZN8nunchaku7kernels13invoke_kernelINS0_9GEMM_W4A4INS0_15GEMMConfig_W4A4ILb1ELb0EEEE30quantize_w4a4_fuse_lora_": "linear W4A4: activation quantization",
    "_ZN2at6native13reduce_kernelILi128ELi4ENS0_8ReduceOpIfNS0_14func_wrapper_tIfZNS0_11sum_functorIfffEclERNS_14Te": "linear W4A4: ordered low-rank sum",
    "_Z41fmha_cutlassF_bf16_aligned_64x128_rf_sm80N22PyTorchMemEffAttention15AttentionKernelIN7cutlass10bfloat16_tE": "cross-attention",
    "triton_red_fused__scaled_dot_product_efficient_attention__to_copy_add_mean_mul_permute_pow_rsqrt_view_11": "cross-attention",
    "triton_red_fused__to_copy_add_mul_native_layer_norm_slice_split_view_w4a4_gelu_mlp_ordered_12": "norms, modulation, residuals",
    "triton_poi_fused_add_mul_split_view_13": "norms, modulation, residuals",
    "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_select_split_unbind_view_3": "norms, modulation, residuals",
}


@pytest.mark.parametrize("name,part", list(KERNELS.items()))
def test_every_kernel_of_the_package_has_its_part(name, part):
    assert bp.part_of(name) == part


def test_runtime_calls_are_no_kernels_and_shares_add_up():
    assert bp.part_of("cudaLaunchKernel") is None and bp.part_of("cuLaunchKernel") is None
    table = bp.parts([("_sum_max", 1.0), ("_rope_apply_kernel", 3.0), ("cudaLaunchKernel", 9.0), ("weird_kernel", 1.0)])
    assert table == [("self-attention: RoPE", 3.0, 0.6), ("self-attention: Sage preparation", 1.0, 0.2),
                     ("other", 1.0, 0.2)]

# Handoff 2026-10-20: KV cache straight into SageAttention 2, then W4A4 in Live

Both done the same day: src/wan_dit_lab/kernels/sage_kv.py (card E25) and the longlive2 preset of live_video.py (card E26). The plan below is kept as the record of the design.

State: block_w4a4_v2 (fused q,k,v and GELU MLP) and float32 RoPE tables are done and tested (cards E23, E24).
Run 19: block 73.8 ms at 704x1280. Nothing committed.

## Task 1 (started, no code yet): FP8 cache -> Sage INT8/FP8 in one pass

Profile (E23): unpacking the FP8 cache to BF16 costs 10.4 ms per block, Sage's own preparation 6.9 ms.
Plan chosen (works at any resolution, same math as now, scales stay dynamic):

- Keep KVCache in FP8 with per-token scales ([B, L, H, D], NHD). Q stays NHD, so no permutes.
- Triton kernels reading two sources per token: t < at from the FP8 cache * scale, t >= at from the BF16 chunk K/V
  (the chunk keeps attending to its own keys unquantized, as now).
  1. K mean over all tokens per head and channel (Sage's smooth_k, km).
  2. K - km to INT8, per-thread groups as Sage's quant_key_per_thread_int8_kernel
     (.venv/Lib/site-packages/sageattention/triton/quant_per_thread.py): per block of 64 tokens 4 scales,
     group g holds tokens 8*i + 2g and 8*i + 2g + 1; k_scale [B, H, nblocks*4].
  3. V max |v| per head and channel, v_scale = max / 2.25 (fp32+fp16 accumulation cap: 448*2.25*64 < 65504).
  4. V / v_scale to FP8, transposed to [B, D, H, Lpad] (NHD convention of per_channel_fp8), Lpad multiple of 64,
     tokens permuted within each 16 by [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15].
- Q: quant_query_per_thread_int8_kernel (q_scale [B, H, ceil(Lq/128)*4*8]), sm_scale = head_dim^-0.5.
- Call sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale,
  k_scale, v_scale, 0 (NHD), 0, 3 (per_thread), sm_scale, 0), as sageattn_qk_int8_pv_fp8_cuda does (core.py 636-824).
- Only for positions "video", no decay, no PC (the turn of ring keys needs BF16); else the current path.
- Test: equal to sageattn on the BF16 unpack within quantization noise; then profile_block (scratch) and a 704 run.
  Expected: about -14 ms per block.

## Task 2: W4A4 v2 + Sage 2 in Live (scripts/live/live_video.py)

Live uses LongLive-2.0-5B (4 steps) with FP8 blocks streamed from RAM (StreamedOnDevice, 12 blocks).
Port: S2 checkpoint (2 steps), pack artifacts/longlive2_5b_s2_blocks_fused.w4a4 via w4a4.mapped_pack and w4a4.slot,
attention "sage". Resolution stays the user's choice (now 480x832).

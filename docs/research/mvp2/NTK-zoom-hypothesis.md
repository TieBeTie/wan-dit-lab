# NTK and the hypothesis of image zoom-in

Date: 04.11.2026. User's observation: NTK sometimes looks like a zoom-in on the original composition with the details kept; this is the desired behavior. The observation does not yet establish either a constant zoom factor or an advantage on all seeds.

## What the launched series checks

512×928; NTK-aware only; M_H = 1.00, 1.05, 1.10, 1.15, 1.20. Scenes in the same order: teacup_pour, astronaut_3_3_4_5, small_fox_snowy_cabin. The large fox is excluded. Seed 0, the same weights, kernels, temporal RoPE, context and sampling. The number of planned videos is 15.

The strength in the config is not M but the exponent t = ln(M_H) / ln(22/16). For width M_W = (40/29)^t. So the maximum multipliers on the two axes differ slightly. At M_H=1 the profile is the identity: the spatial frequencies do not change.

Every run has the same scene order and the same initial seed. The current runtime does not reset the RNG between scenes, so the order is part of the comparison conditions. The previous separate astronaut runs cannot be considered exact repeats of the astronaut after the cup. In the new series the order is fixed; the binary and the hashes of the export sources are saved in the results directory.

Configs: `configs/experiments/director-original/ntk-sweep-100-120-2026-11-04`.
Results: `C:/Users/ramaf/Documents/Experiments/DIT/outputs/quality-search/ntk-sweep-100-120-2026-11-04`.

## NTK is already exponential in the pair number

On each spatial axis there are D=42 real channels, that is 21 pairs. For j=0,...,20:

$$
\theta_j=10000^{-2j/42},\qquad
\theta'_j=\theta_j M^{j/20}
=\theta_j\exp\!\left(\frac{j}{20}\ln M\right).
$$

The multiplier grows linearly on a logarithmic scale, exponentially on an ordinary one. The high frequency j=0 does not change; the lowest one j=20 is multiplied by M. Thus the proposal to check an "exponential profile" is partly already implemented by the current NTK.

Another family, if the result of the current series gives a reason for it:

$$
\theta'_j=\theta_j M^{(j/20)^p},\qquad p>0.
$$

At p=1 this is the current NTK; at p>1 the change is more concentrated near the low frequencies; at 0<p<1 it affects the middle and high pairs earlier. This is our proposal of a parametrization, not an established result of a paper and not an existing config field. At the moment it is not implemented and not run.

## What can be guaranteed mathematically

For fixed features q and k the dependence of a RoPE pair on the distance is determined by the phase theta_j·(p-q). For the replacement of frequencies to correspond exactly to one coordinate transformation p→a·p for all pairs and all distances, it is sufficient and, for all real distances, necessary that:

$$
\theta'_j=a\theta_j\quad\text{for all }j.
$$

This is a uniform scaling of the phases. In NTK at M≠1 the multipliers of different pairs differ. Already j=0 requires a=1, and j=20 requires a=M. Therefore the current NTK is not an exact single scaling of the RoPE coordinates.

Even a uniform scaling of the phases does not prove a geometric zoom of the output image. For that the whole generator must be consistent with the scale transformation: the features, the initial noise, the token grid, the frame borders, the attention normalization and the decoder. A single identity for the Q/K rotations is not enough.

Consequently, a universal guarantee of zoom does not follow from this NTK profile. But a meaningful empirical hypothesis remains: in a range of scales and for some scenes the trained model responds to the profile mainly by changing the scale of the composition, keeping the shape of the objects. The mathematical argument above does not reject this weaker hypothesis.

## What the primary papers give

[FiT, §3.3, VisionNTK / VisionYaRN](https://arxiv.org/html/2402.12376) separates the spatial axes and sets separate scales for H and W. This is close to our implementation. The transfer to other resolutions and aspect ratios is studied; a guarantee of reproducing the original frame through zoom cannot be derived from this work.

[Lumina-Next, §2.2](https://arxiv.org/html/2406.18583) considers the frequency profile and its change over the denoising steps: the global structure and the local details require different treatment of RoPE. This is the basis for studying the shape of the profile after M is chosen. The paper studies resolution extrapolation; our reduced grid and the two-step distilled model require their own check. "Time-aware" here means denoising time, not the time axis of the video.

## How to tell zoom from a change of composition

For each video first estimate a single scale and shift relative to the control NTK M=1 from the static background details — the station window or the small house. The moving fox and the movement of the astronaut must not determine this fit.

Then assess separately:

1. The residual difference of shape and details after compensating the found scale in the common visible area.
2. The stability of the scale in time: a constant zoom-in or a "breathing" of the composition.
3. The execution of actions, the preservation of object identity and artifacts — they do not reduce to the similarity to the control frame.

An arbitrary independent warp for each frame must not be fitted: this could hide exactly the deformations we want to detect. High similarity after one simple geometric transformation will support the zoom hypothesis, but it will not be a universal guarantee and not a standalone proof of quality.

## Parameters of the Live2 config (NTK M=1,10)

File `configs/live2.json`: a copy of `configs/live.json` in which only the RoPE parameters are taken from `ntk-sweep-100-120-2026-11-04/m110.json`.

Comparison of Live 1 and Live 2: of all parameters only three differ, everything else (model, video size 512x928, steps, kernels, offloading) is the same.

| Parameter | Live 1 (`live.json`) | Live 2 (`live2.json`) | What it changes |
|---|---|---|---|
| `rope.stretch` | 1,0 | 0,2992905106483072 | the exponent t for the multiplier M = 1,10: the positions are stretched less |
| `rope.ntk_aware` | false | true | NTK scaling of the RoPE frequencies is enabled |
| `output` | `artifacts/live/512x928-sage-cross` | `artifacts/live/512x928-sage-cross-ntk110` | a separate package directory, so as not to overwrite Live 1 |

The exponent `stretch` is related to the multiplier by the formula t = ln(M_H) / ln(22/16) from the section above; the packages are copied from the Live directory, and if the hash does not match the export is rebuilt.

### All parameters of `configs/live2.json`

The config consists of two components: `export` (package build) and `generate` (Live itself).

**export, package build**

| Parameter | Value | What it means |
|---|---|---|
| `compiler` | `torch_aot_inductor` | the packages are compiled by AOTInductor, Python does not take part in the work |
| `rebuild` | `if_missing_or_changed` | rebuild if the packages are missing or the config changed |
| `output` | `artifacts/live/512x928-sage-cross-ntk110` | the package directory of Live2 |
| `model.architecture` | `Wan2.2-TI2V-5B/config.json` | the Wan 2.2 architecture, 5 billion parameters |
| `model.checkpoint` | `LongLive-2.0-5B-S2/model_te.pt` | the LongLive 2.0 weights |
| `model.block_weights` | `longlive2_5b_s2_blocks_fused.w4a4` | the block weights, 4 bits for weights and activations |
| `model.text_encoder` | T5 (UMT5-XXL), `precision: int8`, weights `t5_int8_fast.bin` | the text encoder in int8 |
| `model.decoder` | `taew2_2.pth` | the TAEHV decoder instead of the full VAE |
| `video.size`, `video.fps` | 512x928, 24 | the output video size and frames per second |
| `sampling` | UniPC, 2 steps, `timestep_shift` 5,0 | the diffusion steps |
| `context.chunk_frames` | 8 | frames in a chunk |
| `context.window_frames` | 32 | the attention window in frames |
| `context.sink_frames` | 8 | frames that always stay in the context (sink) |
| `rope.interpolate` | true | the positions are recomputed from the native size to the video size |
| `rope.native_size` | 704x1280 | the native size of the model, the interpolation goes from it |
| `rope.scene_offset_frames` | 8 | the shift of positions at a scene change |
| `rope.stretch` | 0,2992905106483072 | the exponent t for M = 1,10 |
| `rope.ntk_aware` | true | NTK scaling of the frequencies |
| `rope.by_parts` | null | block-wise interpolation is off |
| `kernels.attention` | `sage_sm89` | Sage attention for SM 8.9 cards |
| `kernels.rope`, `rope_dtype` | `longlive_triton`, float32 | the RoPE kernel and its precision |
| `kernels.fuse` | `rope_into_sage`, `qkv`, `gelu_mlp`: all true | RoPE is fused with attention, QKV and MLP with GELU are fused |
| `kernels.cross_attention` | `sage_sm89` | cross-attention also through Sage |
| `kernels.linear` | `nunchaku_w4a4` | linear layers in 4 bits |
| `kernels.kv_cache`, `kv_store` | `fp8_e4m3fn`, `triton` | the KV cache in FP8 and the kernel writing into it |

**generate, Live itself**

| Parameter | Value | What it means |
|---|---|---|
| `packages_from` | `export` | take the packages from the component above |
| `mode` | `live` | interactive mode, not a run by a script |
| `debug` | `fps` | print the speed in frames per second |
| `drain.every` | 5s | the period of the queue drain (by the name, I did not check more precisely) |
| `prompts.source`, `initial` | `interactive`, «A lone oak tree on a grassy hill sways in the wind» | the prompts come manually, the initial prompt is about an oak |
| `seed` | 0 | the generator seed |
| `offloading.enabled` | true | the weights and cache are offloaded between memory and disk |
| `offloading.calibration` | `if_needed` | calibration only if needed |
| `offloading.resources` | `block_weights`: auto, `kv_cache`: auto | where to keep the block weights and the KV cache is decided by the program |
| `copy_spare` | 1,1 | a margin on the copy time |
| `vram_margin_gib` | 1,5 | the video memory margin, GiB |
| `t5_activations_gib` | 0,33 | memory for the T5 activations, GiB |
| `t5.resident_gib`, `prefetch_tolerance` | 1,0; 0,0 | how much of T5 to keep in video memory and the prefetch tolerance |
| `prefetch_fps_tolerance` | 0,5 | the acceptable speed loss on prefetch |
| `ram_margin_fraction` | 0,0 | the share of RAM kept as a margin |

The difference from m110: in m110 T5 was in bfloat16, attention without RoPE fusion, cross-attention `efficient` and an offline mode with a script; in Live2 all of this stayed as in Live, only `stretch` and `ntk_aware` are taken from m110.

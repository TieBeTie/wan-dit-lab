# BF16 through the native pipeline

`configs/experiments/director-original/s2-bf16-native.json` selects S2, two UniPC
steps, 704×1280, TAEHV, dense BF16 DiT and T5 weights, BF16 KV, math SDPA and
LongLive's original FP32 Triton RoPE. It is a quality control configuration;
it is not a claim of bitwise equivalence to the original Python pipeline.

Both weight formats use the existing byte-row scheduler, two transfer slots,
prefetch streams and calibration. Cache allocation, copies and calibration
estimates use the selected element size. No second runtime or scheduler was
introduced. The four cache tensors retain their existing ABI; BF16 attention
ignores the scale tensors. Dtype branches disappear during package export.

Math SDPA tiles queries in groups of 128 while retaining the full key/value
context. This bounds each score matrix without shortening attention history.
The LongLive RoPE adapter imports the reference kernel rather than copying it.
Its source participates in export identity so edits invalidate old packages.

T5 precision is explicit under `model.text_encoder.precision`: `bfloat16` or
`int8`. Existing configurations without the field keep their INT8 path.

Preparation creates missing immutable BF16 packs before calculating export
identity. It checks the combined disk requirement before writing either pack;
existing packs are never overwritten and must match their source checkpoint.
The normal entry point remains:

```powershell
bin/wan.exe --config configs/experiments/director-original/s2-bf16-native.json
```

On 2026-11-02 the preparation check required 20.77 GiB including a 3 GiB reserve,
with 10.24 GiB available. No full BF16 packages, calibration or videos were
generated. Native-resolution memory fit, full-model accuracy and FPS remain
to be measured after disk space is available.

Tests cover dense row rebinding, source mismatch rejection, T5 parameter
preservation, configuration compatibility, bounded SDPA queries, compiled
original RoPE, compiled BF16 blocks and native FP8/BF16 cache transport through
fill, eviction, shot pinning and reset.

## FP8 control preparation, 2026-11-03

`configs/experiments/director-original/s2-fp8-native.proposed.json` selects
FP8 DiT rows with SDPA, original RoPE, BF16 T5 and BF16 KV. It enables
`debug: "fps"` and drains metrics every five seconds. Native `bench --chunks N`
uses the same packages, reports chunk times and memory, and does not write video.

The new optional `vram_margin_fraction` is a fraction of total device memory.
T5 `activations: "auto"` uses a maximum-token compiled-package measurement;
`resident: "auto"` fits rows after reserving measured generation workspace,
T5 activations, two T5 row slots and the configured margin. Existing explicit
GiB settings remain supported. These automatic measurements require a fresh
export; full-model validation is still pending.

Wan2.2 has a 16-pixel VAE stride and a 2x2 latent patch: one token covers
32x32 pixels. The full native host KV allocation includes the 24-frame history
and the current 8-frame chunk, 30 blocks and K/V plus scale buffers:

| Resolution | KV format | Host KV allocation |
|---|---|---:|
| 512x928 | FP8 | 2.628 GiB |
| 704x1280 | FP8 | 4.985 GiB |
| 704x1280 | BF16 | 9.819 GiB |

The native cache uses VRAM or pinned RAM, not SSD backing. FP8 linear weights
do not change BF16 KV size. A native smoke attempt stopped in the physical-RAM
preflight: 9.82 GiB required, 2.60 GiB available. No packages, calibration or
videos for this FP8 recipe have been generated; no FPS estimate is established.

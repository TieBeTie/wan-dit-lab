# Low-resolution schedule comparison, 2026-10-17

User goal: generate about a minute at low resolution with the same prompt and different noise schedules, then inspect quality and measure proximity to real-time playback. The user reports that the previous `decay` experiment did not produce a satisfactory result; the optimistic assessment in the earlier handoff is not an accepted quality verdict.

## Formula and scope

[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis, section 5.3.2, Eq. 23](https://arxiv.org/html/2403.03206#S5.SS3.SSS2) maps a noise level at area n to area m:

`sigma_new = a * sigma_old / (1 + (a - 1) * sigma_old)`, `a = sqrt(m / n)`.

Composing this with our checkpoint schedule `sigma = s*u / (1 + (s-1)*u)` gives an effective shift `s_new = s*a`. With unchanged chunk length and spatial compression, the pixel-area ratio equals the spatial token-count ratio. The total length of a streamed video is not the number of simultaneously denoised frames and is not included in this ratio.

For 480x832 -> 256x448, `s_new = 5*sqrt((256*448)/(480*832))`, approximately 2.68. This is an experimental transfer of an image-model resolution heuristic to an already distilled four-step video checkpoint, not a validated adaptation or fine-tuning procedure. Changing the schedule alone does not reduce the number of model evaluations or guarantee fidelity.

The comparison holds all other settings fixed: Self-Forcing DMD checkpoint, mammoth prompt from `outputs/m-sink10-pc.json`, seed 0, 256x448, 243 latent frames (969 decoded frames, 60.5625 s at 16 fps), four denoising passes plus clean-cache write, window 21, sink 10, window-relative positions, PC recent 1, FP8 weights and KV, cross-cache, cuDNN attention, torch.compile, streaming TAEHV. No prompt switches.

Schedules: checkpoint shift 5; area-adjusted shift; intermediate shift 3.5. These are three schedules on the same runtime. They are not reproductions of three complete papers. LongLive includes training and a different checkpoint; full Echo-Forcing and Future Forcing have not been reproduced here.

## Reproduction and outputs

`scripts/generation/generate_causal.py` now accepts `--shift` and `--resolution-shift`; the latter multiplies the supplied shift by the area factor above. JSON records the effective shift and actual sigma levels. Defaults retain the checkpoint schedule.

The experiment scripts are retained under ignored `artifacts/`:

- `run_resolution_schedule_comparison.py`: sequential subprocesses, logs and completion status.
- `assemble_resolution_schedule_comparison.py`: labelled side-by-side video, contact sheet and timings.

Outputs under ignored `outputs/` are prefixed `mammoths-256p-60s-schedule-`. The montage is `mammoths-256p-60s-schedules-comparison.mp4`; the contact sheet samples 0, 5, 15, 30, 45 and 60 seconds. Completion status: `lowres-schedules-status.json`; summary: `lowres-schedules-summary.json`.

Decoded RGB chunks are written directly to MP4 and latent chunks are transferred to CPU, preventing minute-long output accumulation on the GPU. Decode timings include device-to-host transfer and MP4 writing. Compare steady-state time after eight warm-up chunks; setup and compilation are separate from playback throughput.

Validation before generation: existing suite 58 passed; new resolution schedule composition test 1 passed; Ruff checks passed for modified Python sources and the new test. Video quality must be assessed from the generated results; no schedule is declared best in advance.

## Completed results

All three runs exited with code 0. Each source video was decoded and checked: 969 frames at 256x448. All saved latent tensors have shape `(16, 243, 32, 56)` and contain only finite values.

| Schedule | Effective shift | Chunk p50 / p95, s | Decode + write p50, s | Steady wall seconds per video second | Peak allocated VRAM, GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 5.0 | 1.24 / 1.28 | 0.049 | 1.715 | 2.73 |
| Area-adjusted | 2.6794565 | 1.27 / 1.304 | 0.058 | 1.768 | 2.73 |
| Intermediate | 3.5 | 1.26 / 1.284 | 0.058 | 1.760 | 2.73 |

The speed target of roughly 1–2 seconds of generation per second of video is met in steady state; cold text encoding and compilation are excluded. The first denoiser phase, including setup/compilation and streaming decoding/writing, took 131 s; the other two took 119.5 and 124.4 s. Peak allocated VRAM is a PyTorch metric, not total device/process memory.

Visual inspection of the contact sheet only (0, 5, 15, 30, 45, 60 s): the original schedule ends in a close-up, whereas the area-adjusted schedule retains the herd in a wider view. The intermediate schedule shows pronounced brightening and background degradation. All three already depict bison-like animals rather than recognizable woolly mammoths near the start, so none establishes good prompt fidelity. These sparse frames do not assess motion smoothness or flicker. The user should judge the complete montage; schedule adaptation is not a demonstrated fix for long-horizon drift.

## BF16 repeat at the user's request

Repeated all three runs with BF16 DiT blocks and BF16 KV storage (`--on-gpu`, without `--fp8` and `--kv-fp8`). TAEHV remains in its existing FP16 decoding path. The assembler checks that prompts, seed, resolution, duration, window, sink, PC, attention backend, compilation, decoder and actual sigma levels match the corresponding FP8 runs. No runtime source changes were needed for this repeat.

Reproduce with `artifacts/run_resolution_schedule_comparison.py --bf16`, then `artifacts/assemble_resolution_schedule_comparison.py --bf16`. All three exited successfully, have 969 frames (60.5625 s), and finite latents of the expected shape. The assembled video was independently checked: 969 frames, 16 fps, 1344x288.

| Shift | Chunk p50 / p95, s | Decode + write p50, s | Steady wall seconds per video second | Peak allocated VRAM, GiB |
| --- | ---: | ---: | ---: | ---: |
| 5.0 | 1.32 / 1.374 | 0.049 | 1.827 | 4.77 |
| 2.6794565 | 1.32 / 1.33 | 0.047 | 1.824 | 4.77 |
| 3.5 | 1.32 / 1.33 | 0.048 | 1.824 | 4.77 |

Files have the suffix `-bf16`; montage: `outputs/mammoths-256p-60s-schedules-bf16-comparison.mp4`; contact sheet: `outputs/mammoths-256p-60s-schedules-bf16-contact.jpg`; metrics: `outputs/lowres-schedules-bf16-summary.json`. Original FP8 outputs are preserved.

Sampled frames still show bison-like animals near the start in BF16. A switch to BF16 does not by itself restore mammoth prompt fidelity in this configuration. This is a visual observation from the contact sheet, not a full motion-quality evaluation.

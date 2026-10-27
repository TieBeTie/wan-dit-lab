"""The presets of Live (once scripts/live/live_video.py, now the Rust host native/live): what differs between the
models it runs, and the Python model of a preset built as Live builds it, for the export of its packages
(scripts/conversion/export_live.py), its Python reference (live_reference.py) and the calibration of W4A4
(convert_w4a4_gptq.py, forcing_kv_heads.py)."""

import copy
import json
from functools import partial

import torch

from wan_dit_lab.generation import causal
from wan_dit_lab.models.checkpoint import LONGLIVE, ROOT, model_config, self_forcing_weights
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.models.positions import rope_at, stretch
from wan_dit_lab.models.rope import rope

KEYS = ["dim", "eps", "ffn_dim", "freq_dim", "in_dim", "num_heads", "num_layers", "out_dim", "text_len"]
POSITIONS = 1 << 16  # latent frames RoPE numbers: at 4 frames each and 16 fps, over an hour of video
SCENE_PREFIX = "The scene transitions. "  # how LongLive 2.0 marks a cut in its prompts (utils/dataset.py)
# what differs between the models; the rest of Live is the same
MODELS = {
    # 2 steps, W4A4 blocks (2.8 GB) and an FP8 cache fit the GPU together; 480x864 (token grid 15x27, the aspect of
    # the trained 22x40) with the RoPE positions of height and width stretched over the trained grid and the shift of
    # training, the recipe of the blind tests E27 and E28
    "longlive2": dict(config=ROOT / "checkpoints" / "Wan2.2-TI2V-5B" / "config.json",
                      checkpoint=ROOT / "checkpoints" / "LongLive-2.0-5B-S2" / "model_te.pt",
                      pack=ROOT / "artifacts" / "longlive2_5b_s2_blocks_fused.w4a4", steps=2, attention="sage",
                      cross_cache=True, taehv=ROOT / "reference" / "taehv" / "taew2_2.pth", fps=24, chunk=8,
                      vae_stride=16, context_frames=32, sink_frames=8, sampler="unipc", native=(704, 1280),
                      # 24 of the 30 W4A4 blocks streamed from pinned RAM (StreamedRows) and no CUDA graph: 5.1 GB on
                      # the card instead of 7.5, 13.0 fps instead of 13.3, room for OBS and a browser (the graph gave
                      # 4% and held 0.5 GB of its own pool)
                      height=480, width=864, kv_fp8=True, stream_blocks=24, rope_dtype="float32",
                      rope_interpolate=True, area_shift=False, fast_text=True, cuda_graph=False),
    "longlive2_fp8": dict(config=ROOT / "checkpoints" / "Wan2.2-TI2V-5B" / "config.json",
                      checkpoint=ROOT / "checkpoints" / "LongLive-2.0-5B" / "model_bf16.pt",
                      taehv=ROOT / "reference" / "taehv" / "taew2_2.pth", fps=24, chunk=8, vae_stride=16,
                      context_frames=32, sink_frames=8, sampler="unipc", native=(704, 1280), height=480, width=832,
                      # 480x832: at 384x672 the old scene comes back after a cut (E16). There the 30 FP8 blocks do
                      # not all fit beside the cache: 12 stream from pinned RAM (8 run out of memory, E17), and the
                      # cache is FP8 (32 frames in bfloat16 do not fit beside the weights)
                      kv_fp8=True, stream_blocks=12, pack=None, steps=None, attention="cudnn", cross_cache=False,
                      rope_dtype="float64", rope_interpolate=False, area_shift=True, fast_text=False, cuda_graph=False),
    "longlive1": dict(config=None, checkpoint=LONGLIVE, taehv=None, fps=16, chunk=3, vae_stride=8, context_frames=12,
                      sink_frames=3, sampler="dmd", native=(480, 832), height=352, width=608, kv_fp8=False,
                      stream_blocks=0, pack=None, steps=None, attention="cudnn", cross_cache=False,
                      rope_dtype="float64", rope_interpolate=False, area_shift=True, fast_text=False, cuda_graph=False),
}


def rope_table(spec, head_dim, device):
    """The RoPE table of `spec` [2, POSITIONS, head_dim / 2]: time, height and width pairs, height and width stretched
    over the trained grid where the preset says so (E27, E28)."""
    d = head_dim
    tables = [rope(POSITIONS, d - 4 * (d // 6)), rope(POSITIONS, 2 * (d // 6)), rope(POSITIONS, 2 * (d // 6))]
    if spec["rope_interpolate"]:
        cell = 2 * spec["vae_stride"]
        scales = stretch([s // cell for s in spec["native"]], [spec["height"] // cell, spec["width"] // cell])
        tables[1:] = [rope_at(POSITIONS, 2 * (d // 6), s) for s in scales]
    return torch.cat(tables, dim=-1).to(device, getattr(torch, spec["rope_dtype"]))


def build_stream(spec, context, device, compiled=True):
    """causal.Stream of `spec` (a W4A4 preset) as Live built it: the pack's blocks, the streamed ones through
    StreamedRows, the blocks and the model's edges under torch.compile unless `compiled` is False."""
    from wan_dit_lab.offloading.offload import StreamedRows
    from wan_dit_lab.quantization import w4a4
    from wan_dit_lab.runtime.compile_all import compile_model_edges

    config = json.loads(spec["config"].read_text(encoding="utf-8")) if spec["config"] else model_config()
    with torch.device("meta"):
        model = WanModel(**{key: config[key] for key in KEYS})
    model.load_state_dict(self_forcing_weights(spec["checkpoint"]), assign=True)
    model.eval()
    rows, layout = w4a4.mapped_pack(model.blocks, spec["pack"], spec["checkpoint"])
    like = copy.deepcopy(model.blocks[0]).to("meta")
    n, k = len(model.blocks), spec["stream_blocks"]
    streamed = [int((j + 0.5) * n / k) for j in range(k)]
    model.blocks = torch.nn.ModuleList([copy.deepcopy(like) if i in streamed else w4a4.slot(row.to(device), like, layout)
                                        for i, row in enumerate(rows)])
    model.block_offloader = StreamedRows(model.blocks, device, streamed, rows, partial(w4a4.slot, like=like, layout=layout))
    model.to(device)
    if compiled:
        forward = torch.compile(type(model.blocks[0]).forward)
        for block in model.block_offloader.modules():
            block.forward = lambda *a, block=block, **kw: forward(block, *a, **kw)
        compile_model_edges(model)
    freqs = rope_table(spec, config["dim"] // config["num_heads"], device)
    return causal.Stream(model, context.to(device), freqs, spec["height"], spec["width"], None,
                         max_frames=spec["context_frames"], sink_frames=spec["sink_frames"], kv_fp8=spec["kv_fp8"],
                         cross_cache=spec["cross_cache"], attention=spec["attention"], shift=causal.SHIFT,
                         chunk=spec["chunk"], vae_stride=spec["vae_stride"], sampler=spec["sampler"],
                         steps=spec["steps"], cuda_graph=False, write_pass=True)

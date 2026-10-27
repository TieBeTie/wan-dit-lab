"""A block_w4a4_v2 pack of LongLive 2.0 S2 whose INT4 residuals are rounded by GPTQ instead of to the nearest
(w4a4.gptq). The inputs X of every layer come from Live itself: Live's W4A4 model at Live's size runs a few prompts
not used in the evaluations, and X^T X of each layer is summed on the GPU, one block at a time (that of the second
layer of the feed forward is 0.8 GB). Smoothing, rank and layout stay those of convert_w4a4.py --fused.

    python scripts/conversion/convert_w4a4_gptq.py --amax configs/calibration/longlive2_s2_act_amax_704.pt
        --output artifacts/longlive2_5b_s2_blocks_fused_gptq.w4a4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import argparse
import copy
import json
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch

import wan_dit_lab.runtime.live_presets as live
from wan_dit_lab.generation import causal
from wan_dit_lab.quantization import w4a4
from wan_dit_lab.models.checkpoint import self_forcing_weights
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.offloading.offload import OnDevice
from wan_dit_lab.models.rope import rope
from wan_dit_lab.models.text_encoder import FastTextEncoder

PROMPTS = [  # from LongLive 2.0's examples, none of them in the evaluations
    "configs/prompts/longlive2_examples.json#raccoon_guitar",
    "configs/prompts/longlive2_examples.json#sheep_river",
]
SLICE = 1024  # tokens at a time through X^T X of the feed forward's hidden layer
EVERY = 4  # of the feed forward's input tokens, one in EVERY is kept for X^T X of its hidden layer (0.8 GB, made after
#   the run: beside Live's model and cache it does not fit 8 GB)


def text(ref):
    path, key = ref.split("#")
    return next(p["text"] for p in json.loads((ROOT / path).read_text(encoding="utf-8"))["prompts"] if p["id"] == key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--amax", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--chunks", type=int, default=6, help="per prompt and block")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=928)
    args = parser.parse_args()
    spec, dev = live.MODELS["longlive2"], torch.device("cuda")
    encoder = FastTextEncoder(dev)
    contexts = [encoder([text(ref)]) for ref in PROMPTS]
    del encoder
    torch.cuda.empty_cache()
    config = json.loads(spec["config"].read_text(encoding="utf-8"))
    weights = self_forcing_weights(spec["checkpoint"])  # mapped from the file, not read into memory
    with torch.device("meta"):
        source = WanModel(**{k: config[k] for k in live.KEYS})
        model = WanModel(**{k: config[k] for k in live.KEYS})
    source.load_state_dict(weights, assign=True)  # the bfloat16 blocks that are quantized
    model.load_state_dict(weights, assign=True)
    model.eval()
    rows, layout = w4a4.mapped_pack(model.blocks, spec["pack"], spec["checkpoint"])
    like = copy.deepcopy(model.blocks[0]).to("meta")
    model.blocks = torch.nn.ModuleList([w4a4.slot(r.to(dev), like, layout) for r in rows])  # Live's model
    model.block_offloader = OnDevice(model.blocks, dev, dtype=None)
    model.to(dev)
    d = config["dim"] // config["num_heads"]
    freqs = torch.cat([rope(live.POSITIONS, d - 4 * (d // 6)), rope(live.POSITIONS, 2 * (d // 6)),
                       rope(live.POSITIONS, 2 * (d // 6))], dim=-1).to(dev, torch.float32)

    def hessians(index):
        """X^T X of every layer group of block `index`, over Live's inputs to it on the calibration prompts."""
        block, H = model.blocks[index], {}

        def add(name, x):
            x = x.reshape(-1, x.shape[-1])
            if name not in H:
                H[name] = torch.zeros(x.shape[1], x.shape[1], device=dev)
            for i in range(0, x.shape[0], SLICE):
                xs = x[i:i + SLICE].float()
                H[name].addmm_(xs.T, xs)

        fc1 = source.blocks[index].ffn[0]

        kept = []

        def ffn(module, inputs):
            x = inputs[0].reshape(-1, inputs[0].shape[-1])
            add("ffn.0", x)
            kept.append(x[::EVERY].to(torch.bfloat16).cpu())

        hooks = [block.ffn.register_forward_pre_hook(ffn)]
        for name in ("self_attn.qkv", "self_attn.o", "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o"):
            hooks.append(block.get_submodule(name).register_forward_pre_hook(
                lambda module, inputs, name=name: add(name, inputs[0])))
        for context in contexts:
            stream = causal.Stream(model, context.to(dev), freqs, args.height, args.width,
                                   torch.Generator(dev).manual_seed(0), max_frames=spec["context_frames"],
                                   sink_frames=spec["sink_frames"], positions="video", kv_fp8=True, cross_cache=True,
                                   attention="sage", shift=causal.SHIFT, chunk=spec["chunk"],
                                   vae_stride=spec["vae_stride"], sampler=spec["sampler"], steps=spec["steps"])
            for _ in range(args.chunks):
                stream.next_chunk()
            del stream
        for hook in hooks:
            hook.remove()
        w1, b1 = fc1.weight.to(dev, torch.bfloat16), fc1.bias.to(dev, torch.bfloat16)
        # the hidden layer is inside the fused kernel: computed again from the bfloat16 fc1 on the kept inputs
        for x in kept:
            x = x.to(dev)
            for i in range(0, x.shape[0], SLICE):
                h = torch.nn.functional.gelu(x[i:i + SLICE] @ w1.T + b1, approximate="tanh")
                add("ffn.2", h.float() + w4a4.GELU_SHIFT)  # the kernel quantizes GELU's output + shift
        del w1, b1, kept
        torch.cuda.empty_cache()
        print(f"block {index}: inputs of {', '.join(sorted(H))} measured", flush=True)
        return H

    amax = torch.load(args.amax)
    print(json.dumps(w4a4.build_pack(source.blocks, amax, args.output, spec["checkpoint"], args.alpha, args.rank,
                                     fused=True, hessians=hessians)), flush=True)

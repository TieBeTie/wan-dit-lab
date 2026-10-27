"""The static heads of Forcing-KV (Ji et al. 2026, Eq. 1) for Live: per block and head, the share of the attention
over the non-sink keys that goes to the chunk itself and the newest finished frame, measured with Live's model at
Live's size on a few prompts once the cache is full; a head is static when its share is above alpha.

    python scripts/experiments/forcing_kv_heads.py --output configs/calibration/forcing_kv_heads_512x928.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import argparse
import copy
import json
import math
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch

import wan_dit_lab.runtime.live_presets as live
from wan_dit_lab.generation import causal
from wan_dit_lab.models.checkpoint import self_forcing_weights
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.models.rope import rope
from wan_dit_lab.models.text_encoder import FastTextEncoder
from wan_dit_lab.offloading.offload import OnDevice
from wan_dit_lab.quantization import w4a4

PROMPTS = [  # three of the screened prompts: a street, water, a person at work
    "configs/prompts/showcase_candidates.json#tokyo_walk",
    "configs/prompts/showcase_candidates.json#waterfall",
    "configs/prompts/showcase_candidates.json#chef_plating",
]


def text(ref):
    path, key = ref.split("#")
    return next(p["text"] for p in json.loads((ROOT / path).read_text(encoding="utf-8"))["prompts"] if p["id"] == key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.8, help="Forcing-KV's threshold of the share")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=928)
    parser.add_argument("--chunks", type=int, default=6, help="per prompt; the cache is full from the 4th")
    parser.add_argument("--query-stride", type=int, default=4, help="every n-th query of the chunk is measured")
    args = parser.parse_args()
    spec, dev = live.MODELS["longlive2"], torch.device("cuda")
    encoder = FastTextEncoder(dev)
    contexts = [encoder([text(ref)]) for ref in PROMPTS]
    del encoder
    config = json.loads(spec["config"].read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = WanModel(**{k: config[k] for k in live.KEYS})
    model.load_state_dict(self_forcing_weights(spec["checkpoint"]), assign=True)
    model.eval()
    rows, layout = w4a4.mapped_pack(model.blocks, spec["pack"], spec["checkpoint"])
    like = copy.deepcopy(model.blocks[0]).to("meta")
    model.blocks = torch.nn.ModuleList([w4a4.slot(r.to(dev), like, layout) for r in rows])
    model.block_offloader = OnDevice(model.blocks, dev, dtype=None)
    model.to(dev)
    d = config["dim"] // config["num_heads"]
    freqs = torch.cat([rope(live.POSITIONS, d - 4 * (d // 6)), rope(live.POSITIONS, 2 * (d // 6)),
                       rope(live.POSITIONS, 2 * (d // 6))], dim=-1).to(dev, torch.float32)

    totals, counts, index = {}, {}, {}
    attend = causal.KVCache.attend

    def measured(self, Q, K, V):
        if self.full():
            b, at, sink, per = index[id(self)], self.saved, self.sink_tokens, self.per_frame
            keys = torch.cat([self.keys(0, at, torch.float32), K.float()], dim=1)[0]  # [tokens, heads, d]
            order = self.frame_of * per + self.place_in_frame
            near = torch.cat([sink + torch.topk(order, per).indices,
                              torch.arange(at, at + K.shape[1], device=K.device)])  # the newest frame and the chunk
            q = Q[0, ::args.query_stride].float()
            share = []
            for h in range(Q.shape[2]):
                p = torch.softmax(q[:, h] @ keys[:, h].T / math.sqrt(Q.shape[-1]), dim=-1)
                share.append(float((p[:, near].sum(-1) / p[:, sink:].sum(-1)).mean()))
            totals[b] = [a + s for a, s in zip(totals.get(b, [0.] * len(share)), share)]
            counts[b] = counts.get(b, 0) + 1
        return attend(self, Q, K, V)

    causal.KVCache.attend = measured
    for context in contexts:
        stream = causal.Stream(model, context.to(dev), freqs, args.height, args.width, torch.Generator(dev).manual_seed(0),
                               max_frames=spec["context_frames"], sink_frames=spec["sink_frames"], positions="video",
                               kv_fp8=True, cross_cache=True, attention="sage", shift=causal.SHIFT, chunk=spec["chunk"],
                               vae_stride=spec["vae_stride"], sampler=spec["sampler"], steps=spec["steps"])
        index = {id(c): i for i, c in enumerate(stream.caches)}
        for _ in range(args.chunks):
            stream.next_chunk()
        print(f"prompt done, {sum(counts.values())} attention calls measured", flush=True)
        del stream
        torch.cuda.empty_cache()
    share = {b: [round(t / counts[b], 4) for t in totals[b]] for b in sorted(totals)}
    static = {b: [h for h, s in enumerate(v) if s > args.alpha] for b, v in share.items()}
    n = sum(len(v) for v in static.values()), sum(len(v) for v in share.values())
    args.output.write_text(json.dumps(dict(
        about="Forcing-KV static heads of LongLive 2.0 S2 with Live's W4A4 pack, Sage over the FP8 cache, "
              f"{args.height}x{args.width}, window {spec['context_frames']}, sink {spec['sink_frames']}: share of the "
              "attention over the non-sink keys on the newest finished frame and the chunk, mean over prompts, steps "
              f"and chunks once the cache is full; static if above alpha",
        alpha=args.alpha, prompts=PROMPTS, static=static, share=share), indent=1), encoding="utf-8")
    print(f"static heads: {n[0]} of {n[1]} ({n[0] / n[1]:.0%})", flush=True)

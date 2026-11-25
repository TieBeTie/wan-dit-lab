"""Compare the live fused block to an independent dense FP32 / FP64-RoPE reference.

This is numerical validation, never a Python video-generation entry point.
"""
import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.conversion.prepare_live import compiler_environment
import os
os.environ.update(compiler_environment())

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.nn.attention import sdpa_kernel, SDPBackend
from wan_dit_lab.models.block import WanBlock
from wan_dit_lab.models.checkpoint import self_forcing_weights
from wan_dit_lab.quantization import w4a4
from wan_dit_lab.runtime.live_pipeline import load_pipeline
from wan_dit_lab.runtime.live_presets import rope_table
from wan_dit_lab.harness.checks import repeated_block_accuracy_gate


class DenseCache:
    def __init__(self, at, buffers):
        self.at, self.buffers = at, buffers

    def attend(self, q, k, v):
        ck, ks, cv, vs = self.buffers
        key = torch.cat([ck[:, :self.at].float() * ks[:, :self.at], k.float()], 1).transpose(1, 2)
        value = torch.cat([cv[:, :self.at].float() * vs[:, :self.at], v.float()], 1).transpose(1, 2)
        q = q.float().transpose(1, 2)
        # Query tiling bounds math-SDPA scratch memory; keys remain complete.
        with sdpa_kernel(SDPBackend.MATH):
            return torch.cat([torch.nn.functional.scaled_dot_product_attention(tile, key, value)
                              for tile in q.split(128, dim=2)], 2).transpose(1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, help="validate native packages instead of a fresh torch.compile")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", nargs="+", type=int, default=[0, 15, 29])
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--repeats", type=int, default=6, help="samples of both paths, including atomic variability")
    parser.add_argument("--contexts", type=Path, default=ROOT / "artifacts/live/480x864/reference.safetensors",
                        help="captured T5 embeddings; their text representation does not depend on video resolution")
    parser.add_argument("--embedding-keys", nargs="+", default=["embedding_A", "embedding_C"])
    parser.add_argument("--input-depth", choices=["native_prefix", "first_block_stress"], default="native_prefix",
                        help="use activations after preceding baseline blocks, or deliberately stress later blocks with first-layer input")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    spec = importlib.util.spec_from_file_location("sage_rope_export_check", ROOT / "scripts/conversion/export_live.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    ex.load_operators()
    recipe, _ = load_pipeline(args.config)
    cfg = json.loads((ROOT / recipe["model"]["architecture"]).read_text())
    h, w = map(int, recipe["video"]["size"].split("x"))
    grid = (recipe["context"]["chunk_frames"], h // 32, w // 32)
    n, heads, d = grid[0] * grid[1] * grid[2], cfg["num_heads"], cfg["dim"] // cfg["num_heads"]
    capacity = n * (recipe["context"]["window_frames"] // grid[0])
    with torch.device("meta"):
        template = WanBlock(cfg["dim"], cfg["ffn_dim"], heads, cfg["eps"])
    checkpoint = ROOT / recipe["model"]["checkpoint"]
    weights = self_forcing_weights(checkpoint)
    rows, layout = w4a4.mapped_pack([template] * cfg["num_layers"], ROOT / recipe["model"]["block_weights"], checkpoint)
    bound = ex.Bound(template, layout, rows.shape[1], attention=recipe["kernels"]["attention"],
                     rope=recipe["kernels"]["rope"], cross_attention=recipe["kernels"]["cross_attention"],
                     rope_into_sage=recipe["kernels"]["fuse"]["rope_into_sage"])
    table64 = rope_table(dict(rope_interpolate=recipe["rope"]["interpolate"], vae_stride=16,
                             native=tuple(map(int, recipe["rope"]["native_size"].split("x"))),
                             height=h, width=w, rope_dtype="float64"), d, "cuda")
    pairs = d // 2 - 2 * (d // 6)
    time64 = table64[:, 3*grid[0]:4*grid[0], :pairs].contiguous()
    hw64 = table64[:, :max(grid), pairs:].contiguous()
    freq64 = torch.cat([torch.nn.functional.pad(time64, (0, 0, 0, hw64.shape[1]-time64.shape[1])), hw64], -1)
    constants = {}
    def load(name, directory=args.baseline):
        kernel = torch._inductor.aoti_load_package(str(directory / f"{name}.pt2"))
        key = str(directory / name)
        constants[key] = load_file(str(directory / f"{name}.safetensors"), device="cuda")
        kernel.load_constants(constants[key], check_full_update=True, user_managed=True)
        return kernel
    kernels = {step: load(f"block_{step}") for step in args.steps}
    pre, text, cross_module = (load(name) for name in ("pre", "text", "cross"))
    candidate_cross_module = load("cross", args.candidate) if args.candidate else ex.Cross(bound)
    candidates = {step: load(f"block_{step}", args.candidate) if args.candidate else
                  torch.compile(ex.BlockStep(bound, step*n, grid, hw64.float()), fullgraph=True, dynamic=False)
                  for step in args.steps}
    with safe_open(args.contexts, framework="pt", device="cpu") as captured:
        context_inputs = [captured.get_tensor(key).cuda().float() for key in args.embedding_keys]
        captured_prompts = json.loads((captured.metadata() or {}).get("prompts", "{}"))
    report = dict(gpu=torch.cuda.get_device_name(), reference="trained dense FP32 block, FP64 RoPE table, math SDPA",
                  inputs="native pre/text/cross packages, seeded latent noise, captured real T5 prompt embeddings",
                  contexts=str(args.contexts), embedding_keys=args.embedding_keys, prompts=captured_prompts,
                  input_depth=args.input_depth,
                  threshold="unchanged 5% relative L2 and dtype-derived max error slack; measured baseline envelope", rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report["passed"] = bool(report["rows"]) and all(row["gate"]["passed"] for row in report["rows"])
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        for block in args.blocks:
            dense = copy.deepcopy(template)
            prefix = f"blocks.{block}."
            dense.load_state_dict({key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}, assign=True)
            dense = dense.cuda().float().eval()
            row = rows[block].cuda()
            for seed_index, seed in enumerate(args.seeds):
                torch.manual_seed(seed)
                latent = torch.randn(cfg["out_dim"],grid[0],h//16,w//16,device="cuda")
                x,e0,_ = pre(latent,torch.tensor([999.],device="cuda"))
                context = text(context_inputs[seed_index % len(context_inputs)])
                cross = cross_module(row,context)
                candidate_cross = candidate_cross_module(row,context)
                reference_cross = dense.cross_attn.keys_values(context.float())
                buffers = [torch.randn(1,capacity,heads,d,device="cuda").to(torch.float8_e4m3fn),
                           torch.rand(1,capacity,heads,1,device="cuda")*.03]
                buffers += [buffers[0].clone(),buffers[1].clone()]
                for step in args.steps:
                    probe_x = x
                    if args.input_depth == "native_prefix":
                        # One forward-pass prefix, no sampler or video generation.
                        # A late layer must see its own activation distribution.
                        for previous in range(block):
                            previous_row = rows[previous].cuda()
                            previous_cross = cross_module(previous_row,context)
                            probe_x = kernels[step](previous_row,probe_x,e0,*previous_cross,time64.float(),
                                                    *[v.clone() for v in buffers])
                    baseline = kernels[step](row,probe_x,e0,*cross,time64.float(),*[v.clone() for v in buffers])
                    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION], set_priority=True):
                        candidate = candidates[step](row,probe_x,e0,*candidate_cross,time64.float(),*[v.clone() for v in buffers])
                    with sdpa_kernel(SDPBackend.MATH):
                        reference = dense(probe_x.float(),e0.float(),grid_sizes=(grid,),freqs=freq64,cache=DenseCache(step*n,buffers),
                                          context_kv=reference_cross)
                    samples = dict(candidate=[candidate], baseline=[baseline])
                    for repeat in range(1, args.repeats):
                        for name, kernel in (("baseline", kernels[step]), ("candidate", candidates[step])):
                            prepared_cross = cross if name == "baseline" else candidate_cross
                            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION], set_priority=True):
                                repeated = kernel(row,probe_x,e0,*prepared_cross,time64.float(),*[v.clone() for v in buffers])
                            samples[name].append(repeated)
                    gate = repeated_block_accuracy_gate(samples["candidate"],samples["baseline"],reference,.05)
                    report["rows"].append(dict(block=block,seed=seed,step=step,gate=gate))
                    save()
                    print(block,seed,step,json.dumps(gate),flush=True)
                    del samples
                    del baseline,candidate,reference
            del dense,row
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

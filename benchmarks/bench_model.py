"""Baseline of the full WanModel: grid (21, 30, 52), FP16 autocast, compiled, two rows in results.jsonl labelled `model_*_compiled`."""

import sys
from pathlib import Path

if "--aot-memory" in sys.argv:
    import json
    import psutil

    memory_report = {"stages": [], "status": "running"}
    memory_path = Path(__file__).resolve().parent / ("aot_reference_memory.json" if "--forward-only" in sys.argv else "aot_memory.json")

    def record_memory(stage):
        info = psutil.Process().memory_info()
        row = {"stage": stage, "private_gib": info.private / 2**30, "working_set_gib": info.rss / 2**30}
        memory_report["stages"].append(row)
        memory_path.write_text(json.dumps(memory_report, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)

    record_memory("python_before_torch")

import torch

if "--aot-memory" in sys.argv:
    record_memory("torch_imported")

from wan_dit_lab.models.checkpoint import load_weights, mapped_weights, model_config
from wan_dit_lab.harness import run
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.models.rope import rope

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference" / "INT8-DiT" / "wan_block"))
from wan.modules.model import WanModel as ReferenceWanModel

KEYS = ["dim", "eps", "ffn_dim", "freq_dim", "in_dim", "num_heads", "num_layers", "out_dim", "text_len"]


def measure(prefix, model, call, settings):
    """Compiled row only, for one model; the model is the only one on the GPU. Eager does not fit: on 32760 tokens
    one block holds about 2 GB besides the 5.3 GB of weights, more than the 8 GB card gives."""
    # AOTInductor is a separate export path, not an argument to torch.compile:
    # 1. exported = torch.export.export(model, example_inputs)  # actual forward inputs
    # 2. torch._inductor.aoti_compile_and_package(exported, package_path="model.pt2")
    # 3. A separate C++ program loads model.pt2 with AOTIModelPackageLoader (LibTorch required).
    # --aot-memory checks this path; export is now blocked by the offloader (see docs/aot-memory-2026-10-16.md).
    # Source: https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_aot_inductor.html
    model.eval()
    # Wrapper: accepts the model's inputs and manages capture, compilation and cached execution.
    # Capture builds a graph of tensor operations and their dependencies on the first call inside run().
    compiled = torch.compile(model)
    run(f"{prefix}_compiled", lambda *a: call(compiled, *a), *ARGS[prefix], **settings)


def probe_aot_memory():
    """One forward at the last video's shape, then try exporting that same offloaded model.

    Private bytes measure host RAM commitments; working set also includes resident mapped weights.
    A successful Python load of a package would still not measure a runtime without Python.
    """
    # This diagnostic takes the torch.export -> AOTInductor path; it does not call measure().
    import traceback
    from wan_dit_lab.offloading.offload import BlockOffloader

    offloader = None
    try:
        record_memory("project_imported")
        memory_report.update(torch=torch.__version__, grid=[9, 30, 52], dtype="bfloat16", forward_passes=1)
        device = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(0.65)  # leave room for the voice tutor
        torch.cuda.init()
        record_memory("cuda_initialized")
        config = model_config()
        with torch.device("meta"):
            model = WanModel(**{key: config[key] for key in KEYS})
        model.load_state_dict(mapped_weights(), assign=True)
        model.eval().requires_grad_(False)
        record_memory("weights_mapped")
        offloader = BlockOffloader(model.blocks, 2, device, ram_window=2)
        model.block_offloader = offloader
        model.to(device)
        torch.cuda.synchronize()
        record_memory("offloader_started")

        grid = (9, 30, 52)  # 33 frames, 480 x 832, matching iceland-motion-480p
        d = config["dim"] // config["num_heads"]
        freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)),
                           rope(1024, 2 * (d // 6))], dim=-1).to(device)
        torch.manual_seed(0)
        args = (torch.randn(1, config["in_dim"], grid[0], grid[1] * 2, grid[2] * 2, device=device),
                torch.tensor([500.0], device=device),
                torch.randn(1, config["text_len"], 4096, device=device), (grid,), freqs)
        record_memory("inputs_created")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            torch.cuda.reset_peak_memory_stats()
            output = model(*args)
            torch.cuda.synchronize()
            memory_report["output_shape"] = list(output.shape)
            memory_report["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
            record_memory("after_forward")
            expected = output.cpu()
            del output
            if "--forward-only" in sys.argv:
                memory_report["status"] = "reference_complete"
                return 0
            print("Trying torch.export on the same model", flush=True)
            exported = torch.export.export(model, args, strict=False)
            record_memory("after_export")
            # A captured graph must reproduce the original, including the changing block weights.
            torch.testing.assert_close(exported.module()(*args).cpu(), expected)
            print("Exported output matches the original", flush=True)
            package = torch._inductor.aoti_compile_and_package(
                exported, package_path=str(memory_path.with_suffix(".pt2")))
            memory_report.update(status="compiled", package=str(package))
    except Exception:
        memory_report.update(status="failed", error=traceback.format_exc())
        print(memory_report["error"], flush=True)
    finally:
        if offloader is not None:
            offloader.stop()
        record_memory("finished")
    return 0 if memory_report["status"] == "compiled" else 1


if __name__ == "__main__" and "--aot-memory" in sys.argv:
    raise SystemExit(probe_aot_memory())


if __name__ == "__main__":
    device = torch.device("cuda")
    config = model_config()
    d = config["dim"] // config["num_heads"]
    grid = (21, 30, 52)
    length = grid[0] * grid[1] * grid[2]  # 32760 tokens
    torch.manual_seed(0)
    x = torch.randn(config["in_dim"], grid[0], grid[1] * 2, grid[2] * 2, device=device)  # latent video: channels, frames, height, width
    t = torch.tensor([500.0], device=device)
    context = torch.randn(config["text_len"], 4096, device=device)  # text encoder output
    freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))], dim=-1).to(device)
    sizes = torch.tensor([grid])
    settings = dict(warmup=2, number=3, repeat=2)  # warmup also pays for compilation

    def call_mine(model, x, t, context, freqs, sizes):
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, cache_enabled=False):
            return model(x[None], t, context[None], sizes, freqs)

    def call_reference(model, x, t, context, freqs, sizes):
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, cache_enabled=False):
            return model([x], t, [context], length)

    ARGS = {
        "model_mine": (x, t, context, freqs, sizes),
        "model_reference": (x, t, context, freqs, sizes),
    }

    # One model at a time: float32 weights take 5.3 GB of the 8 GB card.
    with torch.device(device):
        mine = WanModel(**{key: config[key] for key in KEYS})
    load_weights(mine)
    measure("model_mine", mine, call_mine, settings)
    del mine
    torch.cuda.empty_cache()

    with torch.device(device):
        reference = ReferenceWanModel(**{key: config[key] for key in KEYS}, model_type="t2v")
    load_weights(reference)
    measure("model_reference", reference, call_reference, settings)

"""Compile pure Wan computations; the native runner loads each block's weights.

The existing model and Python offloader remain the reference. No Python object,
queue, or background thread is part of an exported computation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import copy
import gc
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/wan_aot"
OUT.mkdir(exist_ok=True)
REPORT = OUT / "report.json"
GRID = (9, 30, 52)
report = {"status": "running", "grid": GRID, "relative_l2_limit": 0.01, "max_abs_limit": 0.25}


def stage(name, **data):
    report.update(stage=name, **data)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(name, data, flush=True)


import torch

from wan_dit_lab.models.checkpoint import checkpoint_dir, mapped_weights, model_config
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.offloading.offload import BlockOffloader
from wan_dit_lab.models.rope import rope


class Inputs(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        for name in ("patch_embedding", "text_embedding", "time_embedding", "time_projection"):
            setattr(self, name, getattr(model, name))

    def forward(self, x, t, context):
        context = self.text_embedding(context)
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        omega = 10000 ** (-torch.arange(128, dtype=torch.float64, device=t.device) / 128)
        theta = t.double()[:, None] * omega[None, :]
        t = self.time_embedding(torch.cat([torch.cos(theta), torch.sin(theta)], dim=1).float())
        e0 = self.time_projection(t).unflatten(1, (6, -1))
        return x, e0, context, t


class FunctionalInputs(torch.nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.__dict__["inputs"] = inputs  # not registered: its parameters would stay constants of the export
        self.names = tuple(name for name, _ in inputs.named_parameters())

    def forward(self, x, t, context, weights):
        return torch.func.functional_call(self.inputs, dict(zip(self.names, weights)), (x, t, context))


class FunctionalBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.__dict__["block"] = block  # not registered: its parameters would stay constants of the export
        self.names = tuple(name for name, _ in block.named_parameters())

    def forward(self, x, e0, context, freqs, weights):
        return torch.func.functional_call(
            self.block, dict(zip(self.names, weights)), (x, e0),
            {"grid_sizes": (GRID,), "freqs": freqs, "context": context})


class Output(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.head

    def forward(self, x, t):
        x = self.head(x, t)
        f, h, w = GRID
        x = x.view(x.shape[0], f, h, w, 1, 2, 2, 16)
        return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(x.shape[0], 16, f, h * 2, w * 2)


def compile_part(name, module, args, require_no_constants=False):
    stage("export_" + name)
    exported = torch.export.export(module, args, strict=False)
    if require_no_constants:
        assert not exported.graph_signature.parameters, exported.graph_signature
        assert not exported.graph_signature.buffers, exported.graph_signature
    stage("compile_" + name, parameters=len(exported.graph_signature.parameters))
    path = OUT / f"{name}.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(path))
    stage("compiled_" + name)
    return torch._inductor.aoti_load_package(str(path))


def raw_tensor(name, tensor):
    tensor.detach().cpu().contiguous().numpy().tofile(OUT / f"{name}.bin")


def compile_runner(source=None, name="wan_aot_runner"):
    torch_root = Path(torch.__file__).parent
    cuda_root = Path(os.environ.get("CUDA_HOME", ROOT / "artifacts/cuda_toolkit"))
    command = ["cl", "/nologo", "/EHsc", "/std:c++17", "/MD", "/O2",
               f"/I{torch_root / 'include'}", f"/I{torch_root / 'include/torch/csrc/api/include'}",
               f"/I{cuda_root / 'include'}",
               str(source or ROOT / "scripts/conversion/wan_aot_runner.cpp"), f"/Fe:{OUT / (name + '.exe')}",
               f"/Fo:{OUT / (name + '.obj')}", "/link", f"/LIBPATH:{torch_root / 'lib'}",
               f"/LIBPATH:{cuda_root / 'lib/x64'}",
               "torch.lib", "torch_cpu.lib", "torch_cuda.lib", "c10.lib", "c10_cuda.lib", "psapi.lib", "cudart.lib"]
    subprocess.run(command, cwd=OUT, check=True)


def prepare():
    import psutil
    stage("gpu_toolchain_smoke", torch=torch.__version__)
    torch.cuda.set_per_process_memory_fraction(0.65)
    with torch.no_grad():
        smoke = torch.nn.ReLU().cuda().eval()
        value = torch.randn(2, 2, device="cuda")
        loaded = compile_part("gpu_smoke", smoke, (value,))
        torch.testing.assert_close(loaded(value), smoke(value))
        del loaded, smoke, value
    stage("load_reference")
    config = model_config()
    keys = ("dim", "eps", "ffn_dim", "freq_dim", "in_dim", "num_heads", "num_layers", "out_dim", "text_len")
    with torch.device("meta"):
        model = WanModel(**{name: config[name] for name in keys})
    model.load_state_dict(mapped_weights(), assign=True)
    model.eval().requires_grad_(False)
    offloader = BlockOffloader(model.blocks, 2, torch.device("cuda"), ram_window=2)
    model.block_offloader = offloader
    model.to(torch.device("cuda"))
    torch.manual_seed(0)
    x = torch.randn(1, 16, GRID[0], GRID[1] * 2, GRID[2] * 2, device="cuda")
    t = torch.tensor([500.0], device="cuda")
    context = torch.randn(1, config["text_len"], 4096, device="cuda")
    d = config["dim"] // config["num_heads"]
    freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)),
                       rope(1024, 2 * (d // 6))], dim=-1).cuda()
    for name, tensor in (("x", x), ("t", t), ("context", context), ("freqs", freqs)):
        raw_tensor(name, tensor)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        stage("reference_forward")
        expected = model(x, t, context, (GRID,), freqs)
        raw_tensor("expected", expected.float())
        info = psutil.Process().memory_info()
        report["python_reference"] = {"private_gib": info.private / 2**30,
                                      "working_set_gib": info.rss / 2**30}
        offloader.stop()
        model.block_offloader = None
        del offloader, expected
        gc.collect()
        torch.cuda.empty_cache()
        inputs = Inputs(model).eval()
        embedded = inputs(x, t, context)
        input_weights = tuple(parameter.detach().clone() for parameter in inputs.parameters())
        functional_inputs = FunctionalInputs(inputs).eval()
        compile_part("inputs", functional_inputs, (x, t, context, input_weights), True)
        tokens, e0, projected_context, time_embedding = embedded
        block = copy.deepcopy(model.blocks[0]).cuda().eval().requires_grad_(False)
        functional = FunctionalBlock(block).eval()
        # Weight tensors are explicit inputs, separate from the module's registered state.
        weights = tuple(parameter.detach().clone() for parameter in block.parameters())
        args = (tokens, e0, projected_context, freqs, weights)
        first = functional(*args)
        compiled_first = compile_part("block_first", functional, args, True)
        first_actual = compiled_first(*args)
        # bf16 kernels fused by inductor round differently from eager ones: a few elements of 21 million differ by
        # more than an elementwise tolerance, so the check is the relative L2 error, as for the whole model
        error = ((first_actual - first).float().norm() / first.float().norm()).item()
        stage("block_first_check", relative_l2=error)
        assert error <= report["relative_l2_limit"], error
        del compiled_first, first_actual
        middle_args = (first, e0, projected_context, freqs, weights)
        compile_part("block", functional, middle_args, True)
        output = Output(model).eval()
        compile_part("output", output, (first, time_embedding))
    write_manifest(model, x, t, context, freqs, functional.names, functional_inputs.names)
    stage("compile_native_runner")
    compile_runner()
    stage("ready_for_native", status="prepared")


def write_manifest(model, x, t, context, freqs, names, input_names):
    checkpoint = checkpoint_dir() / "diffusion_pytorch_model.safetensors"
    with checkpoint.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(size))
    with (OUT / "manifest.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"WAN_AOT_V1\n{checkpoint}\n{len(model.blocks)} {len(names)}\n")
        for name in names:
            entry = header[f"blocks.0.{name}"]
            assert entry["dtype"] == "F32"
            shape = entry["shape"]
            offsets = [8 + size + header[f"blocks.{i}.{name}"]["data_offsets"][0]
                       for i in range(len(model.blocks))]
            stream.write(" ".join(map(str, (len(shape), *shape, *offsets))) + "\n")
        stream.write(f"{len(input_names)}\n")
        for name in input_names:
            entry = header[name]
            assert entry["dtype"] == "F32"
            stream.write(" ".join(map(str, (len(entry["shape"]), *entry["shape"], 8 + size + entry["data_offsets"][0]))) + "\n")
        for tensor in (x, t, context, freqs):
            dtype = 1 if tensor.dtype == torch.float64 else 0
            stream.write(" ".join(map(str, (dtype, tensor.ndim, *tensor.shape))) + "\n")


def compare():
    import numpy as np
    global report
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["fresh_python_reference"] = json.loads((ROOT / "benchmarks/aot_reference_memory.json").read_text())
    expected = torch.from_numpy(np.fromfile(OUT / "expected.bin", dtype=np.float32))
    measurements = []
    for line in (OUT / "native.log").read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            measurements.append(json.loads(line))
    errors = []
    for name in ("native", "native_second"):
        actual = torch.from_numpy(np.fromfile(OUT / f"{name}.bin", dtype=np.float32))
        assert actual.shape == expected.shape
        difference = (actual - expected).double()
        relative_l2 = (difference.norm() / expected.double().norm()).item()
        max_abs = difference.abs().max().item()
        errors.append({"pass": name, "relative_l2": relative_l2, "max_abs": max_abs})
        assert relative_l2 <= report["relative_l2_limit"] and max_abs <= report["max_abs_limit"], errors
    assert measurements and all(not row["python_loaded"] for row in measurements)
    stage("verified", status="complete", comparisons=errors, native_memory=measurements)


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "compare":
            compare()
        else:
            prepare()
    except BaseException:
        import traceback
        stage("failed", status="failed", error=traceback.format_exc())
        raise

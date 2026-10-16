# AOT export and host memory, 2026-10-16

The existing benchmark now has a diagnostic mode: `.venv/Scripts/python.exe -u benchmarks/bench_model.py --aot-memory`. It runs one forward of the real checkpoint with the existing block offloader, then attempts `torch.export` and AOTInductor compilation. The incomplete second argument to `torch.compile` and the missing `load_weights` import were fixed. The model, RoPE implementation and offloader were not changed.

Environment: Windows, Python 3.14.3, PyTorch 2.10.0+cu128, RTX 4060 Laptop 8 GiB. Grid `(9, 30, 52)`, latent input `(1, 16, 9, 60, 104)`, bf16 autocast, no gradients, two pinned buffers and two GPU slots. The grid matches the last 33-frame 480x832 video; inputs are random, and this measures one denoiser forward, not text encoding, CFG sampling or VAE decoding. The voice tutor remained active; the diagnostic limited the PyTorch allocator to 65% of GPU memory.

| Stage | Private committed memory, GiB | Resident working set, GiB |
| --- | ---: | ---: |
| Python before torch | 0.011 | 0.020 |
| torch imported | 1.193 | 0.480 |
| Project and reference imports | 1.744 | 0.715 |
| Weights mapped | 1.788 | 0.719 |
| Offloader started | 3.120 | 1.954 |
| Inputs created | 3.146 | 2.059 |
| After one forward | 5.124 | 3.120 |

Source for the initial run: [aot_initial_failure.json](../artifacts/aot_initial_failure.json), with the traceback in [aot_initial_failure.log](../artifacts/aot_initial_failure.log). Peak allocated GPU memory was 1.175 GiB. Private committed memory is not the same as physical RAM residency. These are stage snapshots, not a sampled host-memory peak. The initial report's `steps: 1` means one forward, not a complete CFG diffusion step.

The forward succeeded. Export failed in `rope_apply`: the grid is read from tensor values with `.tolist()`, so export encountered a data-dependent expression while reshaping RoPE tensors. The traceback also passes through modules supplied by the offloader that are not registered as submodules; successful export of that mechanism has not been established. No AOT package or native inference executable was produced, and no inference-memory saving has been measured.

A separate initial CPU smoke test exported a 2x2 ReLU successfully, but AOTInductor compilation failed with `Compiler: cl is not found.` At that time `vswhere -products '*' -format json` returned no installed products. WSL Ubuntu has g++ and CMake but no PyTorch installed. The compiler was subsequently installed as described below; Linux PyTorch was not installed.

An independent native-library loading probe used Windows `LoadLibraryEx` in a fresh PowerShell process, with flags `0x1100`, loading the installed `torch_cpu.dll`, `torch_cuda.dll`, then `torch.dll`. It verified that neither a Python DLL nor `torch_python.dll` was loaded. Source: [libtorch_native_memory.json](../benchmarks/libtorch_native_memory.json).

| Native library probe              | Private committed memory, GiB | Resident working set, GiB |
| --------------------------------- | ----------------------------: | ------------------------: |
| PowerShell and measurement helper |                         0.066 |                     0.145 |
| CPU library loaded                |                         0.144 |                     0.230 |
| CUDA library loaded               |                         0.692 |                     0.435 |

Native library loading added 0.626 GiB of private committed memory without Python. This was library initialization only: no CUDA context, model execution or equivalence check was performed, so it is not a replacement for the missing native-inference measurement. It establishes that removing Python alone does not remove these native libraries' memory use.

The [official AOTInductor documentation](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_aot_inductor.html) describes a C++ runner linked to LibTorch. A runtime with neither Python nor LibTorch is a different implementation from that documented AOTInductor route. This experiment does not establish whether rewriting the runtime would save enough RAM to justify it.

## Compiler installation and shape-error fix

On the user's request, Visual Studio Build Tools 2022 version 17.14.41 was installed in `C:/BuildTools/2022`, with MSVC x64/x86 tools and Windows SDK 26100. The detached [compiler check](../artifacts/check_aoti_compiler.ps1) compiled the 2x2 ReLU into [aoti-cpu-probe.pt2](../artifacts/aoti-cpu-probe.pt2), loaded it from Python, and verified identical output. [Status](../benchmarks/aoti_compiler_status.json): exit code 0; [log](../benchmarks/aoti_compiler_check.log). This validates the CPU toolchain, not native inference of the Wan model.

The shape error was fixed by allowing `grid_sizes` to be a tuple of integer triples in `rope_apply` and in the model's output reshape; the diagnostic now supplies that tuple. Existing callers supplying a tensor keep their previous behavior. [test_rope_export.py](../tests/test_rope_export.py) verifies exact equality with the tensor-grid path and exact equality of the exported graph on two different inputs: one test passed in 8.19 s; [log](../benchmarks/rope_export_check.log).

The real-model retry completed its forward, with 5.118 GiB private committed memory and a 3.831 GiB working set afterward. The original data-dependent-shape error is absent. Direct export now fails while tracing a block supplied by the offloader: the traceback starts with `module is not installed as a submodule`, then encounters FakeTensor state errors. Source: [retry data](../benchmarks/aot_memory.json), [traceback](../benchmarks/aot_memory.log), [worker status](../benchmarks/aot_model_status.json).

The offloader's GPU slots are dynamic copies of registered model blocks, and its Python queues, worker thread and CUDA event operations run inside the forward path. Merely removing the shape error has not made that path exportable. No Wan AOT artifact was produced. The diagnostic also now checks exported output against the original before allowing compilation, to avoid accepting an incorrectly captured set of block weights.

"""Cheap package preparation entry point used by wan.exe; exits before Rust loads LibTorch."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wan_dit_lab.runtime.live_pipeline import export_identity, load_pipeline, prepare  # noqa: E402


def compiler_environment():
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    if os.name == "nt" and not shutil.which("cl"):
        candidates = [Path("C:/BuildTools/2022/VC/Auxiliary/Build/vcvars64.bat")]
        installer = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
        if installer.exists():
            found = subprocess.run([str(installer), "-latest", "-products", "*", "-find", "VC/Auxiliary/Build/vcvars64.bat"],
                                   capture_output=True, text=True, check=True)
            candidates = [Path(p) for p in found.stdout.splitlines()] + candidates
        vcvars = next((p for p in candidates if p.exists()), None)
        if vcvars is None:
            raise RuntimeError("export needs MSVC: start from a Developer PowerShell or install C++ Build Tools")
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        result = subprocess.run(f'"{comspec}" /d /s /c "call "{vcvars}" >nul && set"',
                                capture_output=True, text=True, check=True)
        env.update((key.upper(), value) for key, value in
                   (line.split("=", 1) for line in result.stdout.splitlines() if "=" in line and not line.startswith("=")))
    cuda = env.setdefault("CUDA_HOME", str(ROOT / "artifacts/cuda-12.8"))
    env["PATH"] = os.pathsep.join([str(ROOT / ".venv/Scripts"), str(Path(cuda) / "bin"), env.get("PATH", "")])
    env["LINK"] = env.get("LINK", "") + f' /LIBPATH:"{Path(cuda) / "lib/x64"}" cudart.lib'
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dense-weights-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    export, _ = load_pipeline(args.config)
    if args.dense_weights_only:
        from wan_dit_lab.quantization.weight_pack import prepare_dense_weights
        prepare_dense_weights(export, ROOT)
        return
    model = export["model"]
    if ((export["kernels"]["linear"] == "torch_bf16" and not (ROOT / model["block_weights"]).exists())
            or (model["text_encoder"].get("precision", "int8") == "bfloat16"
                and not (ROOT / model["text_encoder"]["block_weights"]).exists())):
        if export["rebuild"] == "never":
            raise ValueError("dense weight packs are missing and rebuild=never")
        print("preparing dense packs in an isolated worker", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", str(args.config), "--dense-weights-only"],
                       cwd=ROOT, env=dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1"), check=True)
    print("checking export input identity", flush=True)
    identity = export_identity(export)
    export["output"] = str(ROOT / export["output"])
    def compile_packages():
        print("compiling packages (one compiler worker)", flush=True)
        env = compiler_environment()
        subprocess.run([sys.executable, str(ROOT / "scripts/conversion/export_live.py"), "--config", str(args.config)],
                       cwd=ROOT, env=env, check=True)
    rebuilt = prepare(export, identity, compile_packages)
    print("packages exported" if rebuilt else "packages match the config; export skipped", flush=True)


if __name__ == "__main__":
    main()

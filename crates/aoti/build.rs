//! Builds shim/aoti_shim.cpp into aoti_shim.dll with MSVC, linked to LibTorch (LIBTORCH) and CUDA (CUDA_HOME). The
//! host loads the DLL at run time, after LibTorch's directory is on the DLL search path, so that the executable
//! itself imports nothing of LibTorch.
use std::{env, path::PathBuf, process::Command};

fn main() {
    let torch = PathBuf::from(env::var("LIBTORCH").expect("LIBTORCH: the torch directory of the Python environment"));
    let cuda = PathBuf::from(env::var("CUDA_HOME").expect("CUDA_HOME"));
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let source = PathBuf::from("shim/aoti_shim.cpp");
    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    let cl = cc::Build::new().cpp(true).target("x86_64-pc-windows-msvc").host("x86_64-pc-windows-msvc")
        .opt_level(2).get_compiler();
    let dll = out.join("aoti_shim.dll");
    let mut command = Command::new(cl.path());
    for (key, value) in cl.get_envs() {
        command.env(key, value);
    }
    let include = torch.join("include");
    command
        .args(["/nologo", "/LD", "/O2", "/MD", "/EHsc", "/std:c++20", "/Zc:__cplusplus", "/permissive-"])
        .args(["/wd4251", "/wd4275", "/wd4244", "/wd4267", "/wd4819", "/DNOMINMAX", "/DUSE_CUDA"])
        .arg(format!("/I{}", include.display()))
        .arg(format!("/I{}", include.join("torch/csrc/api/include").display()))
        .arg(format!("/I{}", cuda.join("include").display()))
        .arg(&source)
        .arg(format!("/Fo{}\\", out.display()))
        .arg(format!("/Fe{}", dll.display()))
        .arg("/link")
        .arg(format!("/LIBPATH:{}", torch.join("lib").display()))
        .arg(format!("/LIBPATH:{}", cuda.join("lib/x64").display()))
        .args(["c10.lib", "c10_cuda.lib", "torch_cpu.lib", "torch_cuda.lib", "torch.lib", "cudart.lib"]);
    let status = command.status().expect("cl");
    assert!(status.success(), "building the AOTInductor shim failed");
    println!("cargo:rustc-env=AOTI_SHIM={}", dll.display());
    println!("cargo:rustc-env=AOTI_TORCH_LIB={}", torch.join("lib").display());
}

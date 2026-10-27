//! The packages of export_live.py with their weights: <name>.pt2 and <name>.safetensors, its constants, which stay
//! out of the package and live in device memory owned here for as long as the package.

use std::path::Path;

use aoti::{DType, Package, Runtime};
use safetensors::{Dtype, SafeTensors};

use crate::gpu::{DeviceBuffer, Stream, DEVICE};

pub struct Loaded {
    pub package: Package,
    _weights: Vec<DeviceBuffer>,
}

impl std::ops::Deref for Loaded {
    type Target = Package;
    fn deref(&self) -> &Package {
        &self.package
    }
}

fn dtype(d: Dtype) -> DType {
    match d {
        Dtype::U8 => DType::U8,
        Dtype::I8 => DType::I8,
        Dtype::I32 => DType::I32,
        Dtype::I64 => DType::I64,
        Dtype::F16 => DType::F16,
        Dtype::BF16 => DType::BF16,
        Dtype::F32 => DType::F32,
        Dtype::F64 => DType::F64,
        Dtype::BOOL => DType::Bool,
        Dtype::F8_E4M3 => DType::F8E4M3,
        other => panic!("a weight of type {other:?}"),
    }
}

pub fn load(runtime: &Runtime, directory: &Path, name: &str, stream: &Stream) -> crate::video::Result<Loaded> {
    let package = runtime.package(&directory.join(format!("{name}.pt2")), DEVICE)?;
    let file = std::fs::File::open(directory.join(format!("{name}.safetensors")))?;
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&map)?;
    let mut weights = Vec::new();
    let mut constants = Vec::new();
    for want in package.constant_names()? {
        let view = tensors.tensor(&want).map_err(|e| format!("{name}.safetensors: {want}: {e}"))?;
        let buffer = DeviceBuffer::new(view.data().len())?;
        stream.copy_to_device(buffer.ptr, view.data())?;
        let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
        constants.push((want, buffer.tensor(0, dtype(view.dtype()), &shape)));
        weights.push(buffer);
    }
    stream.synchronize()?; // the mapping is read before it is dropped
    package.load_constants(&constants)?;
    Ok(Loaded { package, _weights: weights })
}

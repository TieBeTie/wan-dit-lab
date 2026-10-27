//! AOTInductor packages run from Rust: aoti_shim.dll (shim/aoti_shim.cpp) over AOTIModelPackageLoader, loaded at
//! run time after LibTorch's directory is put on the DLL search path, as Python's torch does with
//! os.add_dll_directory.
//!
//! Inputs are descriptors of memory the caller owns ([`Tensor`]); a package's outputs are tensors of LibTorch's
//! allocator, alive until their [`Output`] is dropped.

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const MAX_DIMS: usize = 8;

/// c10::ScalarType of the tensors that cross the shim.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    U8 = 0,
    I8 = 1,
    I32 = 3,
    I64 = 4,
    F16 = 5,
    F32 = 6,
    F64 = 7,
    Bool = 11,
    BF16 = 15,
    F8E4M3 = 24,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::U8 | DType::I8 | DType::Bool | DType::F8E4M3 => 1,
            DType::F16 | DType::BF16 => 2,
            DType::I32 | DType::F32 => 4,
            DType::I64 | DType::F64 => 8,
        }
    }

    fn from_code(code: i32) -> DType {
        match code {
            0 => DType::U8,
            1 => DType::I8,
            3 => DType::I32,
            4 => DType::I64,
            5 => DType::F16,
            6 => DType::F32,
            7 => DType::F64,
            11 => DType::Bool,
            15 => DType::BF16,
            24 => DType::F8E4M3,
            other => panic!("a package gave a tensor of ScalarType {other}"),
        }
    }
}

/// A tensor as the shim sees it: memory, element type, device (-1: the CPU), sizes and strides in elements.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Tensor {
    pub data: *mut c_void,
    dtype: i32,
    pub device: i32,
    ndim: i32,
    sizes: [i64; MAX_DIMS],
    strides: [i64; MAX_DIMS],
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Tensor {
    /// A contiguous tensor over `data` (device memory of `device`, or host memory with device -1).
    pub fn new(data: *mut c_void, dtype: DType, device: i32, shape: &[i64]) -> Tensor {
        assert!(shape.len() <= MAX_DIMS);
        let mut sizes = [0; MAX_DIMS];
        let mut strides = [0; MAX_DIMS];
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            sizes[i] = shape[i];
            strides[i] = stride;
            stride *= shape[i];
        }
        Tensor { data, dtype: dtype as i32, device, ndim: shape.len() as i32, sizes, strides }
    }

    pub fn dtype(&self) -> DType {
        DType::from_code(self.dtype)
    }

    pub fn shape(&self) -> &[i64] {
        &self.sizes[..self.ndim as usize]
    }

    pub fn strides(&self) -> &[i64] {
        &self.strides[..self.ndim as usize]
    }

    pub fn numel(&self) -> usize {
        self.shape().iter().product::<i64>() as usize
    }

    pub fn bytes(&self) -> usize {
        self.numel() * self.dtype().size()
    }

    pub fn is_contiguous(&self) -> bool {
        let mut stride = 1;
        for (size, s) in self.shape().iter().zip(self.strides()).rev() {
            if *size != 1 && *s != stride {
                return false;
            }
            stride *= size;
        }
        true
    }
}

#[repr(C)]
struct RawOutput {
    tensor: Tensor,
    handle: *mut c_void,
}

type Load = unsafe extern "C" fn(*const c_char, i32, *mut c_char, usize) -> *mut c_void;
type Free = unsafe extern "C" fn(*mut c_void);
type Run = unsafe extern "C" fn(*mut c_void, *const Tensor, i32, *mut c_void, *mut RawOutput, i32, *mut c_char, usize) -> i32;
type Memory = unsafe extern "C" fn(i32, *mut i64, *mut i64);
type ConstantNames = unsafe extern "C" fn(*mut c_void, *mut c_char, usize, *mut c_char, usize) -> i64;
type LoadConstants = unsafe extern "C" fn(*mut c_void, *const *const c_char, *const Tensor, i32, *mut c_char, usize) -> i32;
type EmptyCache = unsafe extern "C" fn();

struct Shim {
    _library: libloading::Library,
    load: Load,
    free: Free,
    run: Run,
    tensor_free: Free,
    memory: Memory,
    empty_cache: EmptyCache,
    constant_names: ConstantNames,
    load_constants: LoadConstants,
    _operators: Vec<libloading::Library>,
}

/// LibTorch in this process, with the shim and the operator libraries loaded.
#[derive(Clone)]
pub struct Runtime(Arc<Shim>);

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(windows)]
fn add_dll_directory(directory: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn SetDefaultDllDirectories(flags: u32) -> i32;
        fn AddDllDirectory(directory: *const u16) -> *mut c_void;
    }
    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x1000;
    let wide: Vec<u16> = directory.as_os_str().encode_wide().chain(Some(0)).collect();
    unsafe {
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if AddDllDirectory(wide.as_ptr()).is_null() {
            return Err(Error(format!("cannot add {} to the DLL search path", directory.display())));
        }
    }
    Ok(())
}

impl Runtime {
    /// The LibTorch the shim was built against (LIBTORCH at build time), unless `torch_lib` names another.
    pub fn default_torch_lib() -> PathBuf {
        PathBuf::from(env!("AOTI_TORCH_LIB"))
    }

    /// Loads LibTorch from `torch_lib`, the shim, then each operator library of `operators` (DLLs that register
    /// TORCH_LIBRARY operators the packages call).
    pub fn load(torch_lib: &Path, operators: &[PathBuf]) -> Result<Runtime> {
        add_dll_directory(torch_lib)?;
        let open = |path: &Path| unsafe {
            libloading::Library::new(path).map_err(|e| Error(format!("loading {}: {e}", path.display())))
        };
        let library = open(Path::new(env!("AOTI_SHIM")))?;
        let operators = operators.iter().map(|p| open(p)).collect::<Result<Vec<_>>>()?;
        unsafe {
            let symbol = |name: &[u8]| -> Result<*const c_void> {
                library
                    .get::<*const c_void>(name)
                    .map(|s| *s)
                    .map_err(|e| Error(format!("aoti_shim.dll: {e}")))
            };
            Ok(Runtime(Arc::new(Shim {
                load: std::mem::transmute::<*const c_void, Load>(symbol(b"aoti_load\0")?),
                free: std::mem::transmute::<*const c_void, Free>(symbol(b"aoti_free\0")?),
                run: std::mem::transmute::<*const c_void, Run>(symbol(b"aoti_run\0")?),
                tensor_free: std::mem::transmute::<*const c_void, Free>(symbol(b"aoti_tensor_free\0")?),
                memory: std::mem::transmute::<*const c_void, Memory>(symbol(b"aoti_memory\0")?),
                empty_cache: std::mem::transmute::<*const c_void, EmptyCache>(symbol(b"aoti_empty_cache\0")?),
                constant_names: std::mem::transmute::<*const c_void, ConstantNames>(symbol(b"aoti_constant_names\0")?),
                load_constants: std::mem::transmute::<*const c_void, LoadConstants>(symbol(b"aoti_load_constants\0")?),
                _library: library,
                _operators: operators,
            })))
        }
    }

    pub fn package(&self, path: &Path, device: i32) -> Result<Package> {
        let c = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let mut error = [0 as c_char; 2048];
        let ptr = unsafe { (self.0.load)(c.as_ptr(), device, error.as_mut_ptr(), error.len()) };
        if ptr.is_null() {
            return Err(Error(format!("{}: {}", path.display(), message(&error))));
        }
        Ok(Package { runtime: self.clone(), ptr, name: path.file_name().unwrap().to_string_lossy().into_owned() })
    }

    /// Bytes allocated and reserved by LibTorch's CUDA allocator on `device`.
    pub fn memory(&self, device: i32) -> (i64, i64) {
        let (mut allocated, mut reserved) = (0, 0);
        unsafe { (self.0.memory)(device, &mut allocated, &mut reserved) };
        (allocated, reserved)
    }

    /// The cached free blocks of LibTorch's CUDA allocator back to the driver.
    pub fn empty_cache(&self) {
        unsafe { (self.0.empty_cache)() }
    }
}

fn message(error: &[c_char]) -> String {
    unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy().into_owned()
}

/// A loaded AOTInductor package (.pt2).
pub struct Package {
    runtime: Runtime,
    ptr: *mut c_void,
    name: String,
}

unsafe impl Send for Package {}
unsafe impl Sync for Package {}

impl Package {
    /// The names of the package's constants (its weights, exported out of it).
    pub fn constant_names(&self) -> Result<Vec<String>> {
        let mut error = [0 as c_char; 2048];
        let mut names = vec![0 as c_char; 1 << 16];
        let mut n = unsafe {
            (self.runtime.0.constant_names)(self.ptr, names.as_mut_ptr(), names.len(), error.as_mut_ptr(), error.len())
        };
        if n >= names.len() as i64 {
            names = vec![0 as c_char; n as usize + 1];
            n = unsafe {
                (self.runtime.0.constant_names)(self.ptr, names.as_mut_ptr(), names.len(), error.as_mut_ptr(),
                                                error.len())
            };
        }
        if n < 0 {
            return Err(Error(format!("{}: {}", self.name, message(&error))));
        }
        Ok(message(&names).lines().map(str::to_owned).collect())
    }

    /// Every constant of the package, over memory the caller keeps alive as long as the package.
    pub fn load_constants(&self, constants: &[(String, Tensor)]) -> Result<()> {
        let names: Vec<CString> = constants.iter().map(|(n, _)| CString::new(n.as_str()).unwrap()).collect();
        let pointers: Vec<*const c_char> = names.iter().map(|n| n.as_ptr()).collect();
        let tensors: Vec<Tensor> = constants.iter().map(|(_, t)| *t).collect();
        let mut error = [0 as c_char; 4096];
        let status = unsafe {
            (self.runtime.0.load_constants)(self.ptr, pointers.as_ptr(), tensors.as_ptr(), tensors.len() as i32,
                                            error.as_mut_ptr(), error.len())
        };
        if status < 0 {
            return Err(Error(format!("{}: {}", self.name, message(&error))));
        }
        Ok(())
    }

    /// Runs the package on `stream` (a cudaStream_t, also made the current stream of this thread while it runs).
    pub fn run(&self, inputs: &[Tensor], stream: *mut c_void) -> Result<Vec<Output>> {
        const CAPACITY: usize = 64;
        let mut raw: Vec<RawOutput> = (0..CAPACITY)
            .map(|_| RawOutput { tensor: Tensor::new(std::ptr::null_mut(), DType::U8, -1, &[]), handle: std::ptr::null_mut() })
            .collect();
        let mut error = [0 as c_char; 4096];
        let n = unsafe {
            (self.runtime.0.run)(self.ptr, inputs.as_ptr(), inputs.len() as i32, stream, raw.as_mut_ptr(),
                                 CAPACITY as i32, error.as_mut_ptr(), error.len())
        };
        if n < 0 {
            return Err(Error(format!("{}: {}", self.name, message(&error))));
        }
        raw.truncate(n as usize);
        let outputs: Vec<Output> =
            raw.into_iter().map(|r| Output { tensor: r.tensor, handle: r.handle, runtime: self.runtime.clone() }).collect();
        // a package reads its inputs with the strides it was compiled on, contiguous ones (export_live.py): an
        // output of other strides would be read wrong by the next package, silently
        if let Some((i, o)) = outputs.iter().enumerate().find(|(_, o)| !o.is_contiguous()) {
            return Err(Error(format!("{}: output {i} is not contiguous (strides {:?})", self.name, o.strides())));
        }
        Ok(outputs)
    }
}

impl Drop for Package {
    fn drop(&mut self) {
        unsafe { (self.runtime.0.free)(self.ptr) }
    }
}

/// An output of a package: a tensor of LibTorch's allocator, freed on drop.
pub struct Output {
    pub tensor: Tensor,
    handle: *mut c_void,
    runtime: Runtime,
}

unsafe impl Send for Output {}

impl std::ops::Deref for Output {
    type Target = Tensor;
    fn deref(&self) -> &Tensor {
        &self.tensor
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        unsafe { (self.runtime.0.tensor_free)(self.handle) }
    }
}

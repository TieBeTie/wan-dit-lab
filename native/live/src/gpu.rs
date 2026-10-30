//! The CUDA memory, streams and events Live owns itself, over cudarc's driver API: device buffers and pinned host
//! buffers of exactly the size asked for (PyTorch's pinned allocator rounds every block up to a power of two: a row of
//! 92 MB took 128 MB), streams that do not synchronize with the legacy default stream, events without timing.
//! LibTorch, inside the packages, uses the same primary context.

use std::ffi::c_void;
use std::sync::Arc;

use aoti::{DType, Tensor};
use cudarc::driver::{result, sys, CudaContext, DriverError};

pub type Result<T> = std::result::Result<T, DriverError>;

pub const DEVICE: i32 = 0;

/// The primary context of the GPU, current on the calling thread after `bind`.
#[derive(Clone)]
pub struct Gpu(Arc<CudaContext>);

impl Gpu {
    pub fn new() -> Result<Gpu> {
        let context = CudaContext::new(DEVICE as usize)?;
        Ok(Gpu(context))
    }

    /// Makes the context current on this thread (every thread that calls CUDA).
    pub fn bind(&self) -> Result<()> {
        self.0.bind_to_thread()
    }

    /// Free and total bytes of the GPU.
    pub fn memory(&self) -> Result<(usize, usize)> {
        result::mem_get_info()
    }
}

/// Device memory of one allocation.
pub struct DeviceBuffer {
    pub ptr: sys::CUdeviceptr,
    pub bytes: usize,
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn new(bytes: usize) -> Result<DeviceBuffer> {
        let ptr = unsafe { result::malloc_sync(bytes.max(1))? };
        Ok(DeviceBuffer { ptr, bytes })
    }

    pub fn zeroed(bytes: usize, stream: &Stream) -> Result<DeviceBuffer> {
        let buffer = DeviceBuffer::new(bytes)?;
        unsafe { result::memset_d8_async(buffer.ptr, 0, bytes, stream.0)? };
        Ok(buffer)
    }

    /// The buffer as a contiguous tensor of `shape` from byte `offset` on.
    pub fn tensor(&self, offset: usize, dtype: DType, shape: &[i64]) -> Tensor {
        let t = Tensor::new((self.ptr as usize + offset) as *mut c_void, dtype, DEVICE, shape);
        assert!(offset + t.bytes() <= self.bytes, "a tensor past the end of its buffer");
        t
    }

    pub fn at(&self, offset: usize) -> sys::CUdeviceptr {
        assert!(offset <= self.bytes);
        self.ptr + offset as u64
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = result::free_sync(self.ptr);
        }
    }
}

/// Page-locked host memory of exactly `bytes`.
pub struct PinnedBuffer {
    ptr: *mut c_void,
    pub bytes: usize,
}

unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    pub fn new(bytes: usize) -> Result<PinnedBuffer> {
        let ptr = unsafe { result::malloc_host(bytes.max(1), 0)? };
        Ok(PinnedBuffer { ptr, bytes })
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.bytes) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.bytes) }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = result::free_host(self.ptr);
        }
    }
}

/// A stream that does not wait for the legacy default stream (nor it for this one).
pub struct Stream(pub sys::CUstream);

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    pub fn new() -> Result<Stream> {
        Ok(Stream(result::stream::create(result::stream::StreamKind::NonBlocking)?))
    }

    pub fn raw(&self) -> *mut c_void {
        self.0 as *mut c_void
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { result::stream::synchronize(self.0) }
    }

    pub fn wait(&self, event: &Event) -> Result<()> {
        unsafe { result::stream::wait_event(self.0, event.0, sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT) }
    }

    pub fn copy_to_device(&self, dst: sys::CUdeviceptr, src: &[u8]) -> Result<()> {
        unsafe { result::memcpy_htod_async(dst, src, self.0) }
    }

    pub fn copy_to_host(&self, dst: &mut [u8], src: sys::CUdeviceptr) -> Result<()> {
        unsafe { result::memcpy_dtoh_async(dst, src, self.0) }
    }

    pub fn copy_on_device(&self, dst: sys::CUdeviceptr, src: sys::CUdeviceptr, bytes: usize) -> Result<()> {
        unsafe { result::memcpy_dtod_async(dst, src, bytes, self.0) }
    }

    /// `rows` rows of `width` bytes, `src_pitch` apart on the device, to `dst` `dst_pitch` apart.
    pub fn copy_rows(&self, dst: sys::CUdeviceptr, dst_pitch: usize, src: sys::CUdeviceptr, src_pitch: usize,
                     width: usize, rows: usize) -> Result<()> {
        let copy = sys::CUDA_MEMCPY2D_st {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcHost: std::ptr::null(),
            srcDevice: src,
            srcArray: std::ptr::null_mut(),
            srcPitch: src_pitch,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstHost: std::ptr::null_mut(),
            dstDevice: dst,
            dstArray: std::ptr::null_mut(),
            dstPitch: dst_pitch,
            WidthInBytes: width,
            Height: rows,
        };
        unsafe { sys::cuMemcpy2DAsync_v2(&copy, self.0).result() }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            let _ = result::stream::destroy(self.0);
        }
    }
}

/// An event without timing, to order two streams.
pub struct Event(sys::CUevent);

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    pub fn new() -> Result<Event> {
        Ok(Event(result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)?))
    }

    /// An event that keeps the time it was reached at, for elapsed_ms.
    pub fn timed() -> Result<Event> {
        Ok(Event(result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)?))
    }

    /// Milliseconds of GPU time from this event to `end`, both timed and reached.
    pub fn elapsed_ms(&self, end: &Event) -> Result<f32> {
        unsafe { result::event::elapsed(self.0, end.0) }
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        unsafe { result::event::record(self.0, stream.0) }
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { result::event::synchronize(self.0) }
    }

    /// Whether the GPU has reached the event, without waiting for it.
    pub fn done(&self) -> Result<bool> {
        match unsafe { sys::cuEventQuery(self.0) } {
            sys::CUresult::CUDA_SUCCESS => Ok(true),
            sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            e => Err(e.result().unwrap_err().into()),
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            let _ = result::event::destroy(self.0);
        }
    }
}

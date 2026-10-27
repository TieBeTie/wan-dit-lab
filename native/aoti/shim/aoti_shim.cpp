// The one C++ piece of the Rust host: a C interface over AOTIModelPackageLoader. Tensors cross it as descriptors of
// memory the host owns (wrapped by at::from_blob, never copied); outputs of a package come back as descriptors with a
// handle that keeps the tensor alive until the host frees it.
#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <cstring>
#include <unordered_map>
#include <exception>
#include <string>
#include <vector>

#define EXPORT extern "C" __declspec(dllexport)

constexpr int MAX_DIMS = 8;

struct TensorDesc {
    void* data;
    int32_t dtype;   // c10::ScalarType
    int32_t device;  // CUDA device index, -1 for the CPU
    int32_t ndim;
    int64_t sizes[MAX_DIMS];
    int64_t strides[MAX_DIMS];
};

struct Output {
    TensorDesc tensor;
    void* handle;  // at::Tensor*, freed by aoti_tensor_free
};

namespace {

void fail(const std::exception& e, char* error, size_t length) {
    if (error && length) {
        std::strncpy(error, e.what(), length - 1);
        error[length - 1] = 0;
    }
}

at::Tensor wrap(const TensorDesc& d) {
    auto options = at::TensorOptions()
                       .dtype(static_cast<c10::ScalarType>(d.dtype))
                       .device(d.device < 0 ? at::Device(at::kCPU) : at::Device(at::kCUDA, d.device));
    return at::from_blob(d.data, at::IntArrayRef(d.sizes, d.ndim), at::IntArrayRef(d.strides, d.ndim), options);
}

TensorDesc describe(const at::Tensor& t) {
    TensorDesc d{};
    d.data = t.data_ptr();
    d.dtype = static_cast<int32_t>(t.scalar_type());
    d.device = t.is_cuda() ? t.get_device() : -1;
    d.ndim = static_cast<int32_t>(t.dim());
    for (int i = 0; i < d.ndim && i < MAX_DIMS; ++i) {
        d.sizes[i] = t.size(i);
        d.strides[i] = t.stride(i);
    }
    return d;
}

}  // namespace

EXPORT void* aoti_load(const char* path, int32_t device, char* error, size_t length) {
    try {
        return new torch::inductor::AOTIModelPackageLoader(path, "model", false, 1, device);
    } catch (const std::exception& e) {
        fail(e, error, length);
        return nullptr;
    }
}

EXPORT void aoti_free(void* package) { delete static_cast<torch::inductor::AOTIModelPackageLoader*>(package); }

// Runs the package on `stream` (a cudaStream_t; the same stream is made the current one of this thread, so that
// the operators it calls through the dispatcher, live::* among them, launch there too). Returns the number of
// outputs, -1 on an error.
EXPORT int32_t aoti_run(void* package, const TensorDesc* inputs, int32_t count, void* stream, Output* outputs,
                        int32_t capacity, char* error, size_t length) {
    try {
        std::vector<at::Tensor> in;
        in.reserve(count);
        for (int32_t i = 0; i < count; ++i) in.push_back(wrap(inputs[i]));
        int device = count ? inputs[0].device : 0;
        c10::cuda::CUDAStreamGuard guard(
            c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream), static_cast<c10::DeviceIndex>(device)));
        auto out = static_cast<torch::inductor::AOTIModelPackageLoader*>(package)->run(in, stream);
        if (static_cast<int32_t>(out.size()) > capacity) throw std::runtime_error("more outputs than room for them");
        for (size_t i = 0; i < out.size(); ++i) {
            outputs[i].tensor = describe(out[i]);
            outputs[i].handle = new at::Tensor(std::move(out[i]));
        }
        return static_cast<int32_t>(out.size());
    } catch (const std::exception& e) {
        fail(e, error, length);
        return -1;
    }
}

EXPORT void aoti_tensor_free(void* handle) { delete static_cast<at::Tensor*>(handle); }

// The names of the package's constants (its weights, kept out of the package), one per line, into `names`.
// Returns the length needed, -1 on an error.
EXPORT int64_t aoti_constant_names(void* package, char* names, size_t capacity, char* error, size_t length) {
    try {
        std::string all;
        for (const auto& name : static_cast<torch::inductor::AOTIModelPackageLoader*>(package)->get_constant_fqns())
            all += name + "\n";
        if (all.size() < capacity) std::memcpy(names, all.c_str(), all.size() + 1);
        return static_cast<int64_t>(all.size());
    } catch (const std::exception& e) {
        fail(e, error, length);
        return -1;
    }
}

// Every constant of the package from memory the host owns and keeps alive (user managed: nothing is copied).
EXPORT int32_t aoti_load_constants(void* package, const char* const* names, const TensorDesc* tensors, int32_t count,
                                   char* error, size_t length) {
    try {
        std::unordered_map<std::string, at::Tensor> constants;
        for (int32_t i = 0; i < count; ++i) constants.emplace(names[i], wrap(tensors[i]));
        static_cast<torch::inductor::AOTIModelPackageLoader*>(package)->load_constants(constants, false, true, true);
        return 0;
    } catch (const std::exception& e) {
        fail(e, error, length);
        return -1;
    }
}

// Gives the cached free blocks of the CUDA allocator back to the driver (torch.cuda.empty_cache).
EXPORT void aoti_empty_cache() { c10::cuda::CUDACachingAllocator::emptyCache(); }

// Allocated and reserved bytes of the CUDA allocator on `device`, for the measurements.
EXPORT void aoti_memory(int32_t device, int64_t* allocated, int64_t* reserved) {
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(static_cast<c10::DeviceIndex>(device));
    *allocated = stats.allocated_bytes[0].current;
    *reserved = stats.reserved_bytes[0].current;
}

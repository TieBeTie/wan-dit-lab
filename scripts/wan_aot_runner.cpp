#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using torch::inductor::AOTIModelPackageLoader;

// A CUDA event without the ATen CUDA headers, which need the cuSPARSE, cuBLAS and cuSOLVER headers too.
struct Event {
    cudaEvent_t event = nullptr;
    Event() { TORCH_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess); }
    ~Event() { if (event) cudaEventDestroy(event); }
    void record(cudaStream_t stream) { TORCH_CHECK(cudaEventRecord(event, stream) == cudaSuccess); }
    void block(cudaStream_t stream) { TORCH_CHECK(cudaStreamWaitEvent(stream, event, 0) == cudaSuccess); }
    void synchronize() { TORCH_CHECK(cudaEventSynchronize(event) == cudaSuccess); }
};

void memory(const std::string& stage) {
    PROCESS_MEMORY_COUNTERS_EX info{};
    info.cb = sizeof(info);
    TORCH_CHECK(GetProcessMemoryInfo(GetCurrentProcess(),
                reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&info), sizeof(info)));
    HMODULE modules[2048];
    DWORD needed = 0;
    bool python = false;
    TORCH_CHECK(EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &needed));
    for (size_t i = 0; i < needed / sizeof(HMODULE); ++i) {
        char name[MAX_PATH]{};
        GetModuleBaseNameA(GetCurrentProcess(), modules[i], name, MAX_PATH);
        std::string text(name);
        if (text.rfind("python", 0) == 0 || text == "torch_python.dll") python = true;
    }
    std::cout << "{\"stage\":\"" << stage << "\",\"private_gib\":"
              << info.PrivateUsage / 1073741824.0 << ",\"working_set_gib\":"
              << info.WorkingSetSize / 1073741824.0 << ",\"peak_commit_gib\":"
              << info.PeakPagefileUsage / 1073741824.0 << ",\"python_loaded\":"
              << (python ? "true" : "false") << "}" << std::endl;
    TORCH_CHECK(!python, "Python DLL was unexpectedly loaded");
}

struct Mapping {
    HANDLE file = INVALID_HANDLE_VALUE, mapping = nullptr;
    void* data = nullptr;
    explicit Mapping(const std::string& path) {
        file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        TORCH_CHECK(file != INVALID_HANDLE_VALUE, "Cannot open checkpoint");
        mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        TORCH_CHECK(mapping, "Cannot map checkpoint");
        data = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        TORCH_CHECK(data, "Cannot access checkpoint mapping");
    }
    ~Mapping() {
        if (data) UnmapViewOfFile(data);
        if (mapping) CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
    }
};

std::vector<int64_t> shape(std::istream& stream) {
    size_t rank;
    stream >> rank;
    std::vector<int64_t> sizes(rank);
    for (auto& size : sizes) stream >> size;
    return sizes;
}

at::Tensor read_input(std::istream& manifest, const fs::path& path) {
    int dtype;
    manifest >> dtype;
    auto sizes = shape(manifest);
    auto tensor = torch::empty(sizes, dtype == 1 ? torch::kFloat64 : torch::kFloat32);
    std::ifstream stream(path, std::ios::binary);
    TORCH_CHECK(stream, "Cannot open input ", path.string());
    stream.read(static_cast<char*>(tensor.data_ptr()), tensor.nbytes());
    TORCH_CHECK(stream, "Incomplete input ", path.string());
    return tensor.to(torch::kCUDA);
}

void save(const fs::path& path, const at::Tensor& result) {
    auto cpu = result.to(torch::kCPU, torch::kFloat32).contiguous();
    std::ofstream stream(path, std::ios::binary);
    stream.write(static_cast<const char*>(cpu.data_ptr()), cpu.nbytes());
    TORCH_CHECK(stream, "Cannot write output");
}

int main(int argc, char** argv) {
    try {
        TORCH_CHECK(argc == 2 || argc == 3, "Pass the directory containing the AOT packages and, optionally, the number of forward passes");
        const size_t passes = argc == 3 ? std::stoul(argv[2]) : 2;
        // The linker drops torch_cuda.lib because nothing here refers to it, and without that library torch
        // reports that it has no CUDA support
        TORCH_CHECK(LoadLibraryA("torch_cuda.dll"), "Cannot load torch_cuda.dll");
        c10::InferenceMode guard;
        const fs::path directory(argv[1]);
        memory("native_start");
        std::ifstream manifest(directory / "manifest.txt");
        std::string magic, checkpoint;
        std::getline(manifest, magic);
        std::getline(manifest, checkpoint);
        TORCH_CHECK(magic == "WAN_AOT_V1", "Invalid manifest");
        size_t blocks, count;
        manifest >> blocks >> count;
        std::vector<std::vector<int64_t>> sizes(count);
        std::vector<std::vector<int64_t>> offsets(count, std::vector<int64_t>(blocks));
        for (size_t p = 0; p < count; ++p) {
            sizes[p] = shape(manifest);
            for (auto& offset : offsets[p]) manifest >> offset;
        }
        size_t input_count;
        manifest >> input_count;
        std::vector<std::vector<int64_t>> input_sizes(input_count);
        std::vector<int64_t> input_offsets(input_count);
        for (size_t p = 0; p < input_count; ++p) {
            input_sizes[p] = shape(manifest);
            manifest >> input_offsets[p];
        }
        auto x = read_input(manifest, directory / "x.bin");
        auto t = read_input(manifest, directory / "t.bin");
        auto context = read_input(manifest, directory / "context.bin");
        auto frequencies = read_input(manifest, directory / "freqs.bin");
        Mapping weights_file(checkpoint);
        std::vector<at::Tensor> input_weights;  // the weights of the embeddings go to the GPU once
        for (size_t p = 0; p < input_count; ++p)
            input_weights.push_back(torch::from_blob(static_cast<char*>(weights_file.data) + input_offsets[p],
                                                     input_sizes[p], torch::kFloat32).to(torch::kCUDA));
        AOTIModelPackageLoader input((directory / "inputs.pt2").string());
        AOTIModelPackageLoader first((directory / "block_first.pt2").string());
        AOTIModelPackageLoader middle((directory / "block.pt2").string());
        AOTIModelPackageLoader output((directory / "output.pt2").string());
        // Two sets of weights on the GPU. A thread reads the weights of the next block from the mapped file into
        // pinned memory and copies them on a second stream while this thread computes the block before it.
        std::vector<int64_t> elements(count), starts(count);
        int64_t total = 0;
        for (size_t p = 0; p < count; ++p) {
            elements[p] = 1;
            for (auto size : sizes[p]) elements[p] *= size;
            starts[p] = total;
            total += elements[p];
        }
        std::vector<std::vector<at::Tensor>> slots(2);
        std::vector<at::Tensor> pinned;
        for (size_t slot = 0; slot < 2; ++slot) {
            pinned.push_back(torch::empty({total}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true)));
            for (const auto& size : sizes)
                slots[slot].push_back(torch::empty(size, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)));
        }
        Event copied[2], done[2];
        auto copy_stream = c10::cuda::getStreamFromPool();
        const size_t steps = passes * blocks;
        std::mutex mutex;
        std::condition_variable changed;
        size_t ready = 0, finished = 0;  // steps whose weights are on the GPU, steps whose compute is enqueued
        std::string failure;
        double memcpy_seconds = 0, wait_seconds = 0;
        std::thread loader([&] {
            try {
                c10::InferenceMode inference;
                for (size_t step = 0; step < steps; ++step) {
                    size_t slot = step % 2, layer = step % blocks;
                    if (step >= 2) {
                        std::unique_lock<std::mutex> lock(mutex);
                        changed.wait(lock, [&] { return finished >= step - 1 || !failure.empty(); });
                        if (!failure.empty()) return;
                        lock.unlock();
                        copied[slot].synchronize();  // the copy that read this pinned buffer has finished
                    }
                    auto copy_started = std::chrono::steady_clock::now();
                    char* target = static_cast<char*>(pinned[slot].data_ptr());
                    for (size_t p = 0; p < count; ++p)
                        std::memcpy(target + starts[p] * 4, static_cast<char*>(weights_file.data) + offsets[p][layer],
                                    elements[p] * 4);
                    memcpy_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_started).count();
                    {
                        c10::cuda::CUDAStreamGuard guard(copy_stream);
                        if (step >= 2) done[slot].block(copy_stream.stream());  // the block that used this slot has been computed
                        for (size_t p = 0; p < count; ++p)
                            slots[slot][p].copy_(pinned[slot].narrow(0, starts[p], elements[p]).view(sizes[p]), true);
                        copied[slot].record(copy_stream.stream());
                    }
                    std::lock_guard<std::mutex> lock(mutex);
                    ready = step + 1;
                    changed.notify_all();
                }
            } catch (const std::exception& error) {
                std::lock_guard<std::mutex> lock(mutex);
                failure = error.what();
                changed.notify_all();
            }
        });
        memory("packages_loaded");
        size_t step = 0;
        for (size_t pass = 0; pass < passes; ++pass) {
            auto started = std::chrono::steady_clock::now();
            std::vector<at::Tensor> input_args{x, t, context};
            input_args.insert(input_args.end(), input_weights.begin(), input_weights.end());
            auto embedded = input.run(input_args);
            auto tokens = embedded[0];
            for (size_t layer = 0; layer < blocks; ++layer, ++step) {
                size_t slot = step % 2;
                auto wait_started = std::chrono::steady_clock::now();
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait(lock, [&] { return ready >= step + 1 || !failure.empty(); });
                    TORCH_CHECK(failure.empty(), "Loader failed: ", failure);
                }
                wait_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_started).count();
                auto current = c10::cuda::getCurrentCUDAStream().stream();
                copied[slot].block(current);
                std::vector<at::Tensor> args{tokens, embedded[1], embedded[2], frequencies};
                args.insert(args.end(), slots[slot].begin(), slots[slot].end());
                tokens = (layer == 0 ? first : middle).run(args)[0];
                done[slot].record(current);
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    finished = step + 1;
                    changed.notify_all();
                }
                if (pass == 0) memory("block_" + std::to_string(layer));
            }
            auto result = output.run({tokens, embedded[3]})[0];
            if (pass < 2) save(directory / (pass == 0 ? "native.bin" : "native_second.bin"), result);
            if (pass < 2) memory(pass == 0 ? "after_first_forward" : "after_second_forward");
            auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::cout << "forward_seconds " << pass << " " << seconds << " host_waited_for_weights " << wait_seconds
                      << " loader_memcpy_total " << memcpy_seconds << std::endl;
        }
        loader.join();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

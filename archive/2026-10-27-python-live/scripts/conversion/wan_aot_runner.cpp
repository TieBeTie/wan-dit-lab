#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/core/CachingHostAllocator.h>
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
#include <memory>
#include <cmath>
#include "resource_plan.h"

namespace fs = std::filesystem;
using torch::inductor::AOTIModelPackageLoader;

// A CUDA event without the ATen CUDA headers, which need the cuSPARSE, cuBLAS and cuSOLVER headers too.
struct Event {
    cudaEvent_t event = nullptr;
    explicit Event(bool timing = false) { TORCH_CHECK(cudaEventCreateWithFlags(&event, timing ? cudaEventDefault : cudaEventDisableTiming) == cudaSuccess); }
    ~Event() { if (event) cudaEventDestroy(event); }
    void record(cudaStream_t stream) { TORCH_CHECK(cudaEventRecord(event, stream) == cudaSuccess); }
    void block(cudaStream_t stream) { TORCH_CHECK(cudaStreamWaitEvent(stream, event, 0) == cudaSuccess); }
    void synchronize() { TORCH_CHECK(cudaEventSynchronize(event) == cudaSuccess); }
};

void memory(const std::string& stage, bool cuda_ready = true) {
    PROCESS_MEMORY_COUNTERS_EX info{};
    info.cb = sizeof(info);
    TORCH_CHECK(GetProcessMemoryInfo(GetCurrentProcess(),
                reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&info), sizeof(info)));
    MEMORYSTATUSEX system_memory{};
    system_memory.dwLength = sizeof(system_memory);
    TORCH_CHECK(GlobalMemoryStatusEx(&system_memory));
    size_t free_vram = 0, total_vram = 0;
    if (cuda_ready) TORCH_CHECK(cudaMemGetInfo(&free_vram, &total_vram) == cudaSuccess);
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
    c10::CachingDeviceAllocator::DeviceStats gpu_stats{};
    if (cuda_ready) gpu_stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    std::cout << "{\"stage\":\"" << stage << "\",\"private_gib\":"
              << info.PrivateUsage / 1073741824.0 << ",\"working_set_gib\":"
              << info.WorkingSetSize / 1073741824.0 << ",\"peak_commit_gib\":"
              << info.PeakPagefileUsage / 1073741824.0 << ",\"python_loaded\":"
              << (python ? "true" : "false")
              << ",\"allocated_vram_gib\":" << gpu_stats.allocated_bytes[0].current / 1073741824.0
              << ",\"reserved_vram_gib\":" << gpu_stats.reserved_bytes[0].current / 1073741824.0
              << ",\"peak_allocated_vram_gib\":" << gpu_stats.allocated_bytes[0].peak / 1073741824.0
              << ",\"peak_reserved_vram_gib\":" << gpu_stats.reserved_bytes[0].peak / 1073741824.0
              << ",\"free_vram_gib\":" << free_vram / 1073741824.0
              << ",\"total_vram_gib\":" << total_vram / 1073741824.0
              << ",\"peak_working_set_gib\":" << info.PeakWorkingSetSize / 1073741824.0
              << ",\"free_ram_gib\":" << system_memory.ullAvailPhys / 1073741824.0
              << ",\"total_ram_gib\":" << system_memory.ullTotalPhys / 1073741824.0
              << "}" << std::endl;
    TORCH_CHECK(!python, "Python DLL was unexpectedly loaded");
}

struct StepTiming {
    Event copy_begin{true}, copy_end{true}, gap_begin{true}, dependency_begin{true}, compute_begin{true}, compute_end{true};
    double pack_ms = 0, host_wait_ms = 0;
};

float elapsed_ms(const Event& begin, const Event& end) {
    float value = 0;
    TORCH_CHECK(cudaEventElapsedTime(&value, begin.event, end.event) == cudaSuccess);
    return value;
}

void CUDART_CB delayed_transfer(void* milliseconds) {
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(*static_cast<double*>(milliseconds)));
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
        TORCH_CHECK(argc >= 2, "Pass package directory [passes] [--window N] [--profile] [--copy-delay-ms X] [--pack-delay-ms X] [--output-dir PATH]");
        size_t passes = 2, window = 2;
        bool profiling = false;
        double copy_delay_ms = 0, pack_delay_ms = 0;
        fs::path output_directory = argv[1];
        int argument = 2;
        if (argument < argc && std::string(argv[argument]).rfind("--", 0) != 0) passes = std::stoul(argv[argument++]);
        while (argument < argc) {
            std::string option = argv[argument++];
            if (option == "--profile") { profiling = true; continue; }
            TORCH_CHECK(argument < argc, "Missing argument for ", option);
            std::string value = argv[argument++];
            if (option == "--window") window = std::stoul(value);
            else if (option == "--copy-delay-ms") copy_delay_ms = std::stod(value);
            else if (option == "--pack-delay-ms") pack_delay_ms = std::stod(value);
            else if (option == "--output-dir") output_directory = value;
            else TORCH_CHECK(false, "Unknown option ", option);
        }
        TORCH_CHECK(passes > 0 && window > 0 && window <= 30, "Invalid passes/window");
        TORCH_CHECK(std::isfinite(copy_delay_ms) && copy_delay_ms >= 0 && std::isfinite(pack_delay_ms) && pack_delay_ms >= 0, "Invalid delay");
        fs::create_directories(output_directory);
        auto sample_memory = [&](const std::string& stage) { if (!profiling) memory(stage); };
        // The linker drops torch_cuda.lib because nothing here refers to it, and without that library torch
        // reports that it has no CUDA support
        TORCH_CHECK(LoadLibraryA("torch_cuda.dll"), "Cannot load torch_cuda.dll");
        c10::InferenceMode guard;
        const fs::path directory(argv[1]);
        memory("native_start", false);
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
        std::unique_ptr<Mapping> weights_file;
        std::vector<at::Tensor> input_weights;
        std::unique_ptr<AOTIModelPackageLoader> input, first, middle, output;
        ResourcePlan plan;
        auto fence = [] { TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess); };
        plan.resource("checkpoint", [&] { weights_file = std::make_unique<Mapping>(checkpoint); },
                      [&] { weights_file.reset(); });
        plan.resource("inputs", [&] {
            for (size_t p = 0; p < input_count; ++p)
                input_weights.push_back(torch::from_blob(static_cast<char*>(weights_file->data) + input_offsets[p],
                                                         input_sizes[p], torch::kFloat32).to(torch::kCUDA));
            input = std::make_unique<AOTIModelPackageLoader>((directory / "inputs.pt2").string());
        }, [&] {
            fence();
            input.reset();
            input_weights.clear();
            x = at::Tensor(); t = at::Tensor(); context = at::Tensor();
            sample_memory("released_inputs");
        });
        plan.resource("first", [&] {
            first = std::make_unique<AOTIModelPackageLoader>((directory / "block_first.pt2").string());
        }, [&] { fence(); first.reset(); });
        plan.resource("middle", [&] {
            middle = std::make_unique<AOTIModelPackageLoader>((directory / "block.pt2").string());
        }, [&] { fence(); middle.reset(); });
        plan.resource("output", [&] {
            output = std::make_unique<AOTIModelPackageLoader>((directory / "output.pt2").string());
        }, [&] { fence(); output.reset(); sample_memory("released_output"); });
        // A bounded window of weights on the GPU. A thread reads upcoming blocks from the mapped file into
        // pinned memory and copies them on a second stream while this thread computes the block before it.
        std::vector<int64_t> elements(count), starts(count);
        int64_t total = 0;
        for (size_t p = 0; p < count; ++p) {
            elements[p] = 1;
            for (auto size : sizes[p]) elements[p] *= size;
            starts[p] = total;
            total += elements[p];
        }
        std::vector<std::vector<at::Tensor>> slots(window);
        std::vector<at::Tensor> pinned;
        std::vector<Event> copied(window), done(window);
        auto copy_stream = c10::cuda::getStreamFromPool();
        const size_t steps = passes * blocks;
        std::vector<StepTiming> timings(profiling ? steps : 0);
        std::mutex mutex;
        std::condition_variable changed;
        size_t ready = 0, finished = 0;
        std::string failure;
        double memcpy_seconds = 0, wait_seconds = 0;
        std::thread loader;
        plan.resource("block_weights", [&] {
            for (size_t slot = 0; slot < window; ++slot) {
                pinned.push_back(torch::empty({total}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true)));
                for (const auto& size : sizes)
                    slots[slot].push_back(torch::empty(size, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)));
            }
            loader = std::thread([&] {
                try {
                    c10::InferenceMode inference;
                    c10::cuda::CUDAGuard device_guard(0);
                    for (size_t step = 0; step < steps; ++step) {
                        {
                            std::lock_guard<std::mutex> lock(mutex);
                            if (!failure.empty()) return;
                        }
                        size_t slot = step % window, layer = step % blocks;
                        if (step >= window) {
                            std::unique_lock<std::mutex> lock(mutex);
                            changed.wait(lock, [&] { return finished >= step - window + 1 || !failure.empty(); });
                            if (!failure.empty()) return;
                            lock.unlock();
                            copied[slot].synchronize();  // the copy that read this pinned buffer has finished
                        }
                        auto copy_started = std::chrono::steady_clock::now();
                        if (pack_delay_ms) std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(pack_delay_ms));
                        char* target = static_cast<char*>(pinned[slot].data_ptr());
                        for (size_t p = 0; p < count; ++p)
                            std::memcpy(target + starts[p] * 4, static_cast<char*>(weights_file->data) + offsets[p][layer],
                                        elements[p] * 4);
                        if (profiling) timings[step].pack_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - copy_started).count();
                        {
                            std::lock_guard<std::mutex> lock(mutex);
                            memcpy_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_started).count();
                        }
                        {
                            c10::cuda::CUDAStreamGuard guard(copy_stream);
                            if (step >= window) done[slot].block(copy_stream.stream());  // the block that used this slot has been computed
                            if (copy_delay_ms) TORCH_CHECK(cudaLaunchHostFunc(copy_stream.stream(), delayed_transfer, &copy_delay_ms) == cudaSuccess);
                            if (profiling) timings[step].copy_begin.record(copy_stream.stream());
                            for (size_t p = 0; p < count; ++p)
                                slots[slot][p].copy_(pinned[slot].narrow(0, starts[p], elements[p]).view(sizes[p]), true);
                            if (profiling) timings[step].copy_end.record(copy_stream.stream());
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
        }, [&] {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (failure.empty()) failure = "Stopping weight prefetch";
                changed.notify_all();
            }
            if (loader.joinable()) loader.join();
            fence();  // no copy or kernel may retain a slot when it is freed
            slots.clear();
            pinned.clear();
            at::getHostAllocator(at::kCUDA)->empty_cache();
            frequencies = at::Tensor();
            sample_memory("released_block_weights");
        });
        std::vector<ResourcePlan::Node> nodes;
        std::vector<at::Tensor> embedded;
        at::Tensor tokens;
        std::chrono::steady_clock::time_point started;
        for (size_t pass = 0; pass < passes; ++pass) {
            nodes.push_back({"embed", {"checkpoint", "inputs"}, [&, pass] {
                started = std::chrono::steady_clock::now();
                std::vector<at::Tensor> input_args{x, t, context};
                input_args.insert(input_args.end(), input_weights.begin(), input_weights.end());
                embedded = input->run(input_args);
                tokens = embedded[0];
                embedded[0] = at::Tensor();  // tokens is the sole owner of this intermediate
            }});
            for (size_t layer = 0; layer < blocks; ++layer) {
                nodes.push_back({"block_" + std::to_string(layer),
                                 {"checkpoint", "block_weights", layer == 0 ? "first" : "middle"},
                                 [&, pass, layer] {
                    const size_t step = pass * blocks + layer, slot = step % window;
                    auto current = c10::cuda::getCurrentCUDAStream().stream();
                    if (profiling) timings[step].gap_begin.record(current);
                    auto wait_started = std::chrono::steady_clock::now();
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        changed.wait(lock, [&] { return ready >= step + 1 || !failure.empty(); });
                        TORCH_CHECK(failure.empty(), "Loader failed: ", failure);
                    }
                    wait_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_started).count();
                    if (profiling) {
                        timings[step].host_wait_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait_started).count();
                        timings[step].dependency_begin.record(current);
                    }
                    copied[slot].block(current);
                    if (profiling) timings[step].compute_begin.record(current);
                    std::vector<at::Tensor> args{tokens, embedded[1], embedded[2], frequencies};
                    args.insert(args.end(), slots[slot].begin(), slots[slot].end());
                    tokens = (layer == 0 ? first : middle)->run(args)[0];
                    if (profiling) timings[step].compute_end.record(current);
                    done[slot].record(current);
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        finished = step + 1;
                        changed.notify_all();
                    }
                    if (pass == 0) sample_memory("block_" + std::to_string(layer));
                }});
            }
            nodes.push_back({"output", {"output"}, [&, pass] {
                auto result = output->run({tokens, embedded[3]})[0];
                if (profiling) {
                    Event completed;
                    completed.record(c10::cuda::getCurrentCUDAStream().stream());
                    completed.synchronize();
                }
                if (pass < 2) save(output_directory / (pass == 0 ? "native.bin" : "native_second.bin"), result);
                if (pass + 1 == passes) save(output_directory / "native_last.bin", result);
                tokens = at::Tensor();
                embedded.clear();
                if (pass < 2) sample_memory(pass == 0 ? "after_first_forward" : "after_second_forward");
                auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                std::lock_guard<std::mutex> lock(mutex);
                std::cout << "forward_seconds " << pass << " " << seconds << " host_waited_for_weights " << wait_seconds
                          << " loader_memcpy_total " << memcpy_seconds << std::endl;
            }});
        }
        plan.run(nodes);
        fence();
        c10::cuda::CUDACachingAllocator::emptyCache();
        memory("all_resources_released");
        if (profiling) {
            for (size_t step = 0; step < steps; ++step) {
                auto& timing = timings[step];
                std::cout << "{\"timing\":true,\"pass\":" << step / blocks
                          << ",\"block\":" << step % blocks << ",\"window\":" << window
                          << ",\"pack_ms\":" << timing.pack_ms
                          << ",\"h2d_ms\":" << elapsed_ms(timing.copy_begin, timing.copy_end)
                          << ",\"compute_ms\":" << elapsed_ms(timing.compute_begin, timing.compute_end)
                          << ",\"gpu_ready_gap_ms\":" << elapsed_ms(timing.gap_begin, timing.compute_begin)
                          << ",\"gpu_dependency_wait_ms\":" << elapsed_ms(timing.dependency_begin, timing.compute_begin)
                          << ",\"host_ready_wait_ms\":" << timing.host_wait_ms << "}" << std::endl;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

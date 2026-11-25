// Bench only, off the Live path: Nunchaku's W4A4 GEMM main loop (gemm_w4a4_kernel, EpilogueDefault, INT4, bf16
// out) instantiated with other tilings than GEMMConfig_W4A4 (BLOCK_M 256, BLOCK_N 128, 8 warps), to see whether
// fewer accumulator registers raise occupancy and tensor pipe use. The inputs are random bytes in the layout of each
// tiling: times and counters only, the products are not checked (another tiling needs repacked weights).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "kernels/zgemm/gemm_w4a4.cuh"

namespace bench {
template<int BM, int BN, int NW>
struct Tile {
    static constexpr int BLOCK_M = BM, BLOCK_N = BN, WARP_SIZE = 32, NUM_WARPS = NW;
    static constexpr int INSN_M = 16, INSN_N = 16, INSN_K = 64;
    static constexpr bool FASTER_I2F = false;
    using half_t = __nv_bfloat16;
    using half2_t = __nv_bfloat162;
};

template<typename Config>
void launch(const torch::Tensor &act, const torch::Tensor &wgt, const torch::Tensor &ascales,
            const torch::Tensor &wscales, torch::Tensor &out, int M, int N, int K) {
    using GEMM = nunchaku::kernels::GEMM_W4A4<Config>;
    using Epilogue = typename GEMM::EpilogueDefault;
    using Kernel = typename GEMM::template gemm_w4a4_kernel<Epilogue, false>;
    TORCH_CHECK(M % GEMM::BLOCK_M == 0 && N % GEMM::BLOCK_N == 0, "M, N must be whole tiles");
    dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);
    bool swap = M > N * 2;
    if (swap) std::swap(grid.x, grid.y);
    auto func = nunchaku::kernels::invoke_kernel<Kernel, const typename GEMM::packed_act_t *,
        const typename GEMM::packed_wgt_t *, const typename GEMM::packed_ascale_t *,
        const typename GEMM::packed_wscale_t *, int, int, int, typename Epilogue::Arguments, bool, bool>;
    func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const typename GEMM::packed_act_t *>(act.data_ptr()),
        reinterpret_cast<const typename GEMM::packed_wgt_t *>(wgt.data_ptr()),
        reinterpret_cast<const typename GEMM::packed_ascale_t *>(ascales.data_ptr()),
        reinterpret_cast<const typename GEMM::packed_wscale_t *>(wscales.data_ptr()),
        M, N, K, typename Epilogue::Arguments{reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), M, N}, swap, false);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run(torch::Tensor act, torch::Tensor wgt, torch::Tensor ascales, torch::Tensor wscales, torch::Tensor out,
         int64_t variant, int64_t M, int64_t N, int64_t K) {
    switch (variant) {
    case 0: launch<Tile<256, 128, 8>>(act, wgt, ascales, wscales, out, M, N, K); break;  // Nunchaku's
    case 1: launch<Tile<256, 64, 8>>(act, wgt, ascales, wscales, out, M, N, K); break;   // half the N tiles per warp
    case 2: launch<Tile<128, 64, 4>>(act, wgt, ascales, wscales, out, M, N, K); break;   // same warp tile, 4 warps
    case 3: launch<Tile<256, 96, 8>>(act, wgt, ascales, wscales, out, M, N, K); break;   // three quarters of the N tiles (WARP_M must stay a multiple of 32)
    default: TORCH_CHECK(false, "variant 0..3");
    }
}
}  // namespace bench

TORCH_LIBRARY(gemm_tile, m) { m.def("run(Tensor act, Tensor wgt, Tensor ascales, Tensor wscales, Tensor(a!) out, int variant, int M, int N, int K) -> ()"); }
TORCH_LIBRARY_IMPL(gemm_tile, CUDA, m) { m.impl("run", &bench::run); }

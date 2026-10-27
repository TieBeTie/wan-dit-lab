// The W4A4 layers of SVDQuant (Nunchaku, reference/nunchaku, Apache 2.0, commit 302e0e9) as operators live::*, so
// that a model exported by AOTInductor calls them from C++, with no Python in the process. The same computation as
// linear and gelu_mlp of quantization/w4a4.py, which now call these operators.
#include "ops.h"

#include <torch/library.h>

namespace {

constexpr int64_t GROUP = 64;  // input channels per INT4 scale
constexpr int64_t PAD = 256;   // rows of the activation, padded to a multiple, as Nunchaku does

int64_t padded(int64_t m) { return (m + PAD - 1) / PAD * PAD; }

std::vector<float> lora_scales(int64_t rank) { return std::vector<float>((rank + 15) / 16, 1.0f); }

struct Quantized {
    at::Tensor q, scales, lora;
};

// x [M, K] bfloat16 to INT4 with its scales, and the down projection of the low-rank branch
Quantized quantize(const at::Tensor& x, const at::Tensor& proj_down, const at::Tensor& smooth) {
    int64_t M = x.size(0), K = x.size(1), Mp = padded(M);
    auto o = x.options();
    Quantized r{at::empty({Mp, K / 2}, o.dtype(at::kByte)), at::empty({K / GROUP, Mp}, o),
                at::empty({Mp, proj_down.size(1)}, o.dtype(at::kFloat))};
    nunchaku::ops::quantize_w4a4_act_fuse_lora(x, r.q, r.scales, proj_down, r.lora, smooth, false, false);
    return r;
}

at::Tensor w4a4_linear(const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& wscales,
                       const at::Tensor& proj_down, const at::Tensor& proj_up, const at::Tensor& smooth,
                       const at::Tensor& bias) {
    auto a = quantize(x, proj_down, smooth);
    auto out = at::empty({x.size(0), proj_up.size(0)}, x.options());
    nunchaku::ops::gemm_w4a4(a.q, qweight, out, {}, a.scales, wscales, {}, {}, a.lora, proj_up, {}, {}, {}, {}, {},
                             bias, {}, {}, {}, false, lora_scales(proj_down.size(1)), false, false, 1.0f, {}, {}, {},
                             {}, 0);
    return out;
}

// GELU(x W1^T + b1) W2^T + b2 (Nunchaku's fused_gelu_mlp): the first GEMM applies GELU, the down projection of the
// second layer's low-rank branch and the INT4 quantization of GELU + shift in its epilogue
at::Tensor w4a4_gelu_mlp(const at::Tensor& x, const at::Tensor& qweight1, const at::Tensor& wscales1,
                         const at::Tensor& proj_down1, const at::Tensor& proj_up1, const at::Tensor& smooth1,
                         const at::Tensor& bias1, const at::Tensor& qweight2, const at::Tensor& wscales2,
                         const at::Tensor& proj_down2, const at::Tensor& proj_up2, const at::Tensor& smooth2,
                         const at::Tensor& bias2) {
    auto a = quantize(x, proj_down1, smooth1);
    int64_t Mp = a.q.size(0), H = proj_up1.size(0);
    auto o = x.options();
    auto qh = at::empty({Mp, H / 2}, o.dtype(at::kByte));
    auto hscales = at::empty({H / GROUP, Mp}, o);
    auto lora_h = at::empty({Mp, proj_down2.size(1)}, o.dtype(at::kFloat));  // zeroed by the kernel
    nunchaku::ops::gemm_w4a4(a.q, qweight1, {}, qh, a.scales, wscales1, hscales, {}, a.lora, proj_up1, proj_down2,
                             lora_h, {}, {}, {}, bias1, smooth2, {}, {}, false, lora_scales(proj_down1.size(1)), false,
                             false, 1.0f, {}, {}, {}, {}, 0);
    auto out = at::empty({x.size(0), proj_up2.size(0)}, o);
    nunchaku::ops::gemm_w4a4(qh, qweight2, out, {}, hscales, wscales2, {}, {}, lora_h, proj_up2, {}, {}, {}, {}, {},
                             bias2, {}, {}, {}, true, lora_scales(proj_down2.size(1)), false, false, 1.0f, {}, {}, {},
                             {}, 0);
    return out;
}

at::Tensor w4a4_linear_meta(const at::Tensor& x, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            const at::Tensor& proj_up, const at::Tensor&, const at::Tensor&) {
    return at::empty({x.size(0), proj_up.size(0)}, x.options());
}

at::Tensor w4a4_gelu_mlp_meta(const at::Tensor& x, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                              const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                              const at::Tensor&, const at::Tensor&, const at::Tensor& proj_up2, const at::Tensor&,
                              const at::Tensor&) {
    return at::empty({x.size(0), proj_up2.size(0)}, x.options());
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(live, m) {
    m.def("w4a4_linear(Tensor x, Tensor qweight, Tensor wscales, Tensor proj_down, Tensor proj_up, Tensor smooth, "
          "Tensor bias) -> Tensor");
    m.def("w4a4_gelu_mlp(Tensor x, Tensor qweight1, Tensor wscales1, Tensor proj_down1, Tensor proj_up1, "
          "Tensor smooth1, Tensor bias1, Tensor qweight2, Tensor wscales2, Tensor proj_down2, Tensor proj_up2, "
          "Tensor smooth2, Tensor bias2) -> Tensor");
}

TORCH_LIBRARY_IMPL(live, CUDA, m) {
    m.impl("w4a4_linear", &w4a4_linear);
    m.impl("w4a4_gelu_mlp", &w4a4_gelu_mlp);
}

TORCH_LIBRARY_IMPL(live, Meta, m) {
    m.impl("w4a4_linear", &w4a4_linear_meta);
    m.impl("w4a4_gelu_mlp", &w4a4_gelu_mlp_meta);
}

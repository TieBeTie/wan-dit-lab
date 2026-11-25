// The sm89 Sage kernel of ops/sage_sm89.cu, dense and over lists of key blocks.
#pragma once
#include <ATen/ATen.h>

void sage_sm89_dense(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
                     const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale, float sm_scale);

void sage_sm89_blocks(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
                      const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale,
                      const at::Tensor& lut, const at::Tensor& valid, float sm_scale);

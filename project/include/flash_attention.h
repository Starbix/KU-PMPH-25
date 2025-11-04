// flash_attention.h
// Header for FlashAttention CUDA implementation with PyTorch interface
#pragma once

#include <torch/torch.h>

namespace flash_attention {

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

torch::Tensor forward_with_params(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                  int B_c, int B_r, int bdim_x, int bdim_y);

}  // namespace flash_attention

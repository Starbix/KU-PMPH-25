// flash_attention.h
// Header for FlashAttention CUDA implementation with PyTorch interface
#pragma once

#include <torch/torch.h>

namespace flash_attention {

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

}  // namespace flash_attention

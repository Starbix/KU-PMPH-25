// flash_attention.h
// Header for FlashAttention CUDA implementation with PyTorch interface
#pragma once

#include <torch/torch.h>

namespace flash_attention {

// Forward function for flash attention mechanism
// Input tensors Q, K, V should have shape (seq_len, head_dim)
// Returns output tensor with same shape as Q
// TODO: Add scaling factor 1/sqrt(head_dim)
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
double forward_duration(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

}  // namespace flash_attention
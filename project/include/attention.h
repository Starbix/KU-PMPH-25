// attention.h
// Header for standard attention CUDA implementation with PyTorch interface
#pragma once

#include <torch/torch.h>

namespace attention {

// Forward function for standard attention mechanism
// Input tensors Q, K, V should have shape (seq_len, head_dim)
// Returns output tensor with same shape as Q
// TODO: Add scaling factor 1/sqrt(head_dim)
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

}  // namespace attention
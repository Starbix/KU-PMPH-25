// flash_attention.cuh
// Header for FlashAttention CUDA implementation
#pragma once

#include <cuda_runtime.h>

namespace flash_attention {

// Forward declarations for the flash attention mechanisms
cudaError_t compute(float* d_q, float* d_k, float* d_v, float* d_output,
                   int batch_size, int seq_len, int head_dim);

}  // namespace flash_attention
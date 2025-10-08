// attention.cuh
// Header for standard attention CUDA implementation
#pragma once

#include <cuda_runtime.h>


namespace attention {

// Forward declarations for the attention mechanisms
cudaError_t fill_ones(float* d_K, int batch_size, int seq_len, int head_dim);

}  // namespace attention
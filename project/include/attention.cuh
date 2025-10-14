// attention.cuh
// Header for standard attention CUDA implementation
#pragma once

#include <cuda_runtime.h>
#include "../src/utils.cu"



namespace attention {

// Forward declarations for the attention mechanisms
cudaError_t fill_ones(float* d_K, int batch_size, int seq_len, int head_dim);

template<class ElTp>
cudaError_t compute_attention(ElTp* Q, ElTp* K, ElTp* V, uint32_t N, uint32_t d, ElTp* O);

}  // namespace attention
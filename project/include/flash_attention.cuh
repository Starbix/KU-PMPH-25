// flash_attention.cuh
// Header for FlashAttention CUDA implementation
#pragma once

#include <cuda_runtime.h>
#include "../src/utils.cu"


namespace flash_attention {

    template<class ElTp>
    cudaError_t compute(ElTp* Q, ElTp* K, ElTp* V, uint32_t N, uint32_t d, ElTp* O);


}  
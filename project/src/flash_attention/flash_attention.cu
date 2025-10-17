#include "flash_attention_kernel.cu.h"
#include "../utils.cu"



namespace flash_attention {


    template<class ElTp>
    cudaError_t compute(ElTp* Q, ElTp* K, ElTp* V, uint32_t N, uint32_t d, ElTp* O) {
        // TODO
        return cudaGetLastError();
    }


    template cudaError_t compute<float>(
        float*, float*, float*, uint32_t, uint32_t, float*);


}
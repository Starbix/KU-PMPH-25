#include <cuda_runtime.h>



namespace utils {
    struct FlashAttentionResult {
        double duration;
        cudaError_t cudaError;
    };
}


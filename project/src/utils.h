#include <cuda_runtime.h>

typedef unsigned int uint32_t;

namespace utils {
    struct FlashAttentionResult {
        double duration;
        cudaError_t cudaError;
    };
}


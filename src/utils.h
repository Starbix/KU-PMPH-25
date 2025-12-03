#include <cuda_runtime.h>

typedef unsigned int uint32_t;

namespace utils {
    struct AttentionResult {
        double duration;
        cudaError_t cudaError;
    };
}


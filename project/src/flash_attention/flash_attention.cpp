#include "../../include/flash_attention.h"
#include <cuda_runtime.h>
#include "../utils.h"

// Forward declaration of CUDA kernel launcher


utils::FlashAttentionResult launch_flash_attention_kernels(
    float* Q_ptr, float* K_ptr, float* V_ptr, float* O_ptr,
    int seq_len, int head_dim
);
utils::FlashAttentionResult launch_flash_attention_kernels_with_params(
    float* Q_ptr, float* K_ptr, float* V_ptr, float* O_ptr,
    int seq_len, int head_dim, int B_c, int B_r, int bdim_x, int bdim_y
);

namespace flash_attention {
    
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Input validation
    TORCH_CHECK(Q.defined(), "Q tensor is not defined");
    TORCH_CHECK(K.defined(), "K tensor is not defined");
    TORCH_CHECK(V.defined(), "V tensor is not defined");
    
    TORCH_CHECK(Q.dim() == 2, "Q must have 2 dimensions (seq_len, head_dim)");
    TORCH_CHECK(K.dim() == 2, "K must have 2 dimensions (seq_len, head_dim)");
    TORCH_CHECK(V.dim() == 2, "V must have 2 dimensions (seq_len, head_dim)");
    
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have the same shape");
    TORCH_CHECK(Q.sizes() == V.sizes(), "Q and V must have the same shape");
    
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA device");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA device");
    TORCH_CHECK(V.device().is_cuda(), "V must be on CUDA device");
    
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be float32");
    
    // Get dimensions
    int seq_len = Q.size(0);
    int head_dim = Q.size(1);
    
    // Ensure tensors are contiguous
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    
    // Create output tensor with same shape as Q
    torch::Tensor O = torch::zeros_like(Q);
    
    // Get raw pointers
    float* Q_ptr = Q.data_ptr<float>();
    float* K_ptr = K.data_ptr<float>();
    float* V_ptr = V.data_ptr<float>();
    float* O_ptr = O.data_ptr<float>();
    
    // Launch CUDA kernels
    utils::FlashAttentionResult result = launch_flash_attention_kernels(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        seq_len, head_dim
    );
    
    TORCH_CHECK(result.cudaError == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(result.cudaError));
    
    return O;
}

torch::Tensor forward_with_parameters(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
int B_c, int B_r, int bdim_x, int bdim_y) {
     // Input validation
    TORCH_CHECK(Q.defined(), "Q tensor is not defined");
    TORCH_CHECK(K.defined(), "K tensor is not defined");
    TORCH_CHECK(V.defined(), "V tensor is not defined");
    
    TORCH_CHECK(Q.dim() == 2, "Q must have 2 dimensions (seq_len, head_dim)");
    TORCH_CHECK(K.dim() == 2, "K must have 2 dimensions (seq_len, head_dim)");
    TORCH_CHECK(V.dim() == 2, "V must have 2 dimensions (seq_len, head_dim)");
    
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have the same shape");
    TORCH_CHECK(Q.sizes() == V.sizes(), "Q and V must have the same shape");
    
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA device");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA device");
    TORCH_CHECK(V.device().is_cuda(), "V must be on CUDA device");
    
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be float32");
    
    // Get dimensions
    int seq_len = Q.size(0);
    int head_dim = Q.size(1);
    
    // Ensure tensors are contiguous
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    
    // Create output tensor with same shape as Q
    torch::Tensor O = torch::zeros_like(Q);
    
    // Get raw pointers
    float* Q_ptr = Q.data_ptr<float>();
    float* K_ptr = K.data_ptr<float>();
    float* V_ptr = V.data_ptr<float>();
    float* O_ptr = O.data_ptr<float>();
    
    // Launch CUDA kernels
    utils::FlashAttentionResult result = launch_flash_attention_kernels_with_params(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        seq_len, head_dim,
        B_c, B_r, bdim_x, bdim_y
    );
    
    TORCH_CHECK(result.cudaError == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(result.cudaError));
    
    return O;
}


// this function can be used for parameter optimization
double forward_duration(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int B_c, int B_r, int bdim_x, int bdim_y) {
    // Input validation
    TORCH_CHECK(Q.defined(), "Q tensor is not defined");
    TORCH_CHECK(K.defined(), "K tensor is not defined");
    TORCH_CHECK(V.defined(), "V tensor is not defined");
    
    TORCH_CHECK(Q.dim() == 2, "Q must have 2 dimensions (seq_len, head_dim)");
    TORCH_CHECK(K.dim() == 2, "K must have 2 dimensions (seq_len, head_dim)");
    TORCH_CHECK(V.dim() == 2, "V must have 2 dimensions (seq_len, head_dim)");
    
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have the same shape");
    TORCH_CHECK(Q.sizes() == V.sizes(), "Q and V must have the same shape");
    
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA device");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA device");
    TORCH_CHECK(V.device().is_cuda(), "V must be on CUDA device");
    
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be float32");
    
    // Get dimensions
    int seq_len = Q.size(0);
    int head_dim = Q.size(1);
    
    // Ensure tensors are contiguous
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    
    // Create output tensor with same shape as Q
    torch::Tensor O = torch::zeros_like(Q);
    
    // Get raw pointers
    float* Q_ptr = Q.data_ptr<float>();
    float* K_ptr = K.data_ptr<float>();
    float* V_ptr = V.data_ptr<float>();
    float* O_ptr = O.data_ptr<float>();
    
    // Launch CUDA kernels
    utils::FlashAttentionResult result = launch_flash_attention_kernels_with_params(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        seq_len, head_dim,
        B_c, B_r, bdim_x, bdim_y
    );
    
    TORCH_CHECK(result.cudaError == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(result.cudaError));
    
    return result.duration;
}

} // namespace flash_attention

// #define ENABLE_PYBIND

// PyBind11 module definition for Python interface
#ifdef ENABLE_PYBIND
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention::forward, "Flash attention forward pass");
    m.def("forward_with_parameters", &flash_attention::forward_with_parameters, "Flash attention forward pass");
    m.def(
        "forward_duration", 
        (double(*)(torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int))
            &flash_attention::forward_duration,
        "Flash attention forward pass duration for optimization"
    );
}
#endif
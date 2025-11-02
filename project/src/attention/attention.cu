#include "attention_kernel.cu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include "../utils.h"


// Simple transpose kernel
__global__ void transpose_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// This function was copied from the lecture notes, chapter 6.2
template<class ElTp, int T>
__global__ void transpose(ElTp* M, ElTp* M_tr, uint32_t rows, uint32_t cols) {
    __shared__ ElTp tile [T][T];
    unsigned int tidx = threadIdx .x;
    unsigned int tidy = threadIdx .y;
    unsigned int j = blockIdx .x*T + tidx ;
    unsigned int i = blockIdx .y*T + tidy ;
    if ( j < cols && i < rows )
        tile [ tidy ][ tidx ] = M[i*cols + j ];
    __syncthreads ();
    if ( j < cols && i < rows )
        M_tr [j*rows + i] = tile [ tidy ][ tidx ];
}


// TODO: remove dummy code
// CUDA kernel to fill the K matrix with 1s
__global__ void fill_ones_kernel(float* K, int total_elements) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within bounds
    if (idx < total_elements) {
        // Set the value to 1.0
        K[idx] = 1.0f;
    }
}

template<class ElTp, int T>
utils::FlashAttentionResult compute(ElTp* Q, ElTp* K, ElTp* V, uint32_t N, uint32_t d, ElTp* O) {
    // implement standard attention by using the kernels from attention_kernel
    auto start = std::chrono::high_resolution_clock::now();
    // Different grid configurations for different operations
    dim3 block(T, T, 1);

    // 1. Transpose K
    int dim_N = ceil(((float)N)/T);
    int dim_d = ceil(((float)d)/T);
    dim3 grid(dim_d, dim_N, 1);

    ElTp* K_tr;
    cudaMalloc(&K_tr, N * d * sizeof(ElTp));
    transpose<ElTp, T> <<<grid, block>>>(K, K_tr, N, d);

    // 2. Call compute_S(Q, K_tr, N, d, S)
    ElTp* S;
    cudaMalloc(&S, N * N * sizeof(ElTp));
    compute_S<ElTp, T> <<<grid, block>>>(Q, K_tr, N, d, S);

    // 3. Call compute_P(S, N)
    ElTp* P;
    cudaMalloc(&P, N * N * sizeof(ElTp));
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    compute_P_online<ElTp> <<<num_blocks, threads_per_block>>>(S, P, N);

    // 4. Call compute_O(V, P, N, d, O)
    compute_O<ElTp, T> <<<grid, block>>>(V, P, N, d, O);

    cudaFree(K_tr);
    cudaFree(S);
    cudaFree(P);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    // Return any error that occurred
    return utils::FlashAttentionResult{.duration = duration, .cudaError = cudaGetLastError()};
}

// Profiling version with CUDA events
cudaError_t compute_with_profiling(float* Q, float* K, float* V, uint32_t N, uint32_t d, float* O) {
    const uint32_t T = 16;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // Different grid configurations for different operations
    dim3 block(T, T, 1);
    
    // 1. Transpose K (N×d → d×N)
    int dim_N = ceil(((float)N)/T);
    int dim_d = ceil(((float)d)/T);
    dim3 grid(dim_d, dim_N, 1);
    
    float* K_tr;
    cudaMalloc(&K_tr, N * d * sizeof(float));
    
    cudaEventRecord(start);
    transpose<float, T> <<<grid, block>>>(K, K_tr, N, d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("    Transpose kernel: %.3f ms\n", milliseconds);

    // 2. Compute S = Q * K^T (N×N result)
    dim3 grid_S(dim_N, dim_N, 1);
    
    float* S;
    cudaMalloc(&S, N * N * sizeof(float));
    
    cudaEventRecord(start);
    compute_S<float, T> <<<grid_S, block>>>(Q, K_tr, N, d, S);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("    compute_S kernel: %.3f ms\n", milliseconds);

    // 3. Compute P = softmax(S) (N×N)
    float* P;
    cudaMalloc(&P, N * N * sizeof(float));
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
    cudaEventRecord(start);
    compute_P_online<float> <<<num_blocks, threads_per_block>>>(S, P, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("    compute_P kernel: %.3f ms\n", milliseconds);

    // 4. Compute O = P * V (N×d result)
    dim3 grid_O(dim_d, dim_N, 1);
    cudaEventRecord(start);
    compute_O<float, T> <<<grid_O, block>>>(V, P, N, d, O);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("    compute_O kernel: %.3f ms\n", milliseconds);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(K_tr);
    cudaFree(S);
    cudaFree(P);

    return cudaGetLastError();
}

template utils::FlashAttentionResult compute<float, 32>(float* Q, float* K, float* V, uint32_t N, uint32_t d, float* O);
template utils::FlashAttentionResult compute<float, 16>(float* Q, float* K, float* V, uint32_t N, uint32_t d, float* O);

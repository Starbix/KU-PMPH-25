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
utils::AttentionResult compute(ElTp* Q, ElTp* K, ElTp* V, uint32_t N, uint32_t d, ElTp* O) {
    auto start = std::chrono::high_resolution_clock::now();

    dim3 block(T, T, 1);
    int dim_N = ceil(((float)N)/T);
    int dim_d = ceil(((float)d)/T);
    dim3 grid(dim_d, dim_N, 1);

    // 1. Transpose K
    ElTp* K_tr;
    cudaMalloc(&K_tr, N * d * sizeof(ElTp));
    transpose<ElTp, T> <<<grid, block>>>(K, K_tr, N, d);

    // 2. Call compute_S(Q, K_tr, N, d, S)
    dim3 grid_S(dim_N, dim_N, 1);
    ElTp* S;
    cudaMalloc(&S, N * N * sizeof(ElTp));
    compute_S<ElTp, T> <<<grid_S, block>>>(Q, K_tr, N, d, S);

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
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    // Return any error that occurred
    return utils::AttentionResult{.duration = duration, .cudaError = cudaGetLastError()};
}


template utils::AttentionResult compute<float, 32>(float* Q, float* K, float* V, uint32_t N, uint32_t d, float* O);
template utils::AttentionResult compute<float, 16>(float* Q, float* K, float* V, uint32_t N, uint32_t d, float* O);

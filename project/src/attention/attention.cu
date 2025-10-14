#include "../../include/attention.cuh"
#include "attention_kernel.cu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "../utils.cu"

namespace attention {

    // This function was copied from the lecture notes, chapter 6.2
    template<class ElTp, int T>
    __global__ void transpose(ElTp* M, ElTp* M_tr, uint32_t rows, uint32_t cols) {
        __shared__ float tile [T][T];
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

    // TODO: remove dummy function
    // Host function to launch the fill_ones_kernel
    cudaError_t fill_ones(float* d_K, int batch_size, int seq_len, int head_dim) {
        // Calculate total elements in K
        int total_elements = batch_size * seq_len * head_dim;
        
        // Configure the kernel launch parameters
        int blockSize = 256;
        int numBlocks = (total_elements + blockSize - 1) / blockSize;
        
        // Launch the kernel
        fill_ones_kernel<<<numBlocks, blockSize>>>(d_K, total_elements);
        
        // Return any error that occurred
        return cudaGetLastError();
    }

    template<class ElTp>
    cudaError_t compute(ElTp* Q, ElTp* K, ElTp* V, uint32_t N, uint32_t d, ElTp* O) {
        // implement standard attention by using the kernels from attention_kernel
        const uint32_t T = 16;
        int  dimy = ceil( ((float)N)/T ); 
        int  dimx = ceil( ((float)d)/T );
        dim3 block(T, T, 1), grid(dimx, dimy, 1);

        // 1. Transpose K
        ElTp* K_tr = nullptr; // TODO
        cudaMalloc(&K_tr, N * d * sizeof(ElTp));
        transpose<ElTp, T> <<<grid, block>>>(K, K_tr, N, d);

        // 2. Call compute_S(Q, K_tr, N, d, S)
        ElTp* S = nullptr; // TODO        
        cudaMalloc(&S, N * N * sizeof(ElTp));
        compute_S<ElTp, T> <<<grid, block>>>(Q, K_tr, N, d, S);

        // 3. Call compute_P(S, N)
        ElTp* P = nullptr;        
        cudaMalloc(&P, N * N * sizeof(ElTp));
        int threads_per_block = 256;         
        int num_blocks = (N + threads_per_block - 1) / threads_per_block;
        compute_P_online<ElTp> <<<num_blocks, threads_per_block>>>(S, P, N);

        // 4. Call compute_O(V, P, N, d, O)
        compute_S<ElTp, T> <<<grid, block>>>(V, P, N, d, O);

        cudaFree(K_tr);
        cudaFree(S);
        cudaFree(P);

        // Return any error that occurred
        return cudaGetLastError();
    }

    
    template cudaError_t compute<float>(
        float*, float*, float*, uint32_t, uint32_t, float*);
}

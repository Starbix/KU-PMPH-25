#include "attention_kernel.cu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "../utils.cu"


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

    // Return any error that occurred
    return cudaGetLastError();
}


// CUDA kernel launcher that interfaces with the torch wrapper
cudaError_t launch_attention_kernels(
    float* Q_ptr, float* K_ptr, float* V_ptr, float* O_ptr,
    int batch_heads, int seq_len, int head_dim
) {
    // Process each batch_head independently
    for (int batch_head = 0; batch_head < batch_heads; batch_head++) {
        // Calculate offsets for current batch_head
        size_t offset = batch_head * seq_len * head_dim;
        float* Q_batch = Q_ptr + offset;
        float* K_batch = K_ptr + offset;
        float* V_batch = V_ptr + offset;
        float* O_batch = O_ptr + offset;

        // Allocate temporary memory for S (attention scores) and P (probabilities)
        float* S_dev;
        float* P_dev;
        float* K_tr_dev; // K transpose

        size_t S_size = seq_len * seq_len * sizeof(float);
        size_t K_tr_size = seq_len * head_dim * sizeof(float);

        cudaError_t err = cudaMalloc(&S_dev, S_size);
        if (err != cudaSuccess) return err;

        err = cudaMalloc(&P_dev, S_size);
        if (err != cudaSuccess) {
            cudaFree(S_dev);
            return err;
        }

        err = cudaMalloc(&K_tr_dev, K_tr_size);
        if (err != cudaSuccess) {
            cudaFree(S_dev);
            cudaFree(P_dev);
            return err;
        }

        // Transpose K for matrix multiplication
        // Simple transpose kernel - could be optimized
        dim3 transpose_block(16, 16);
        dim3 transpose_grid((head_dim + 15) / 16, (seq_len + 15) / 16);

        // Launch transpose kernel
        transpose_kernel<<<transpose_grid, transpose_block>>>(
            K_batch, K_tr_dev, seq_len, head_dim
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(S_dev);
            cudaFree(P_dev);
            cudaFree(K_tr_dev);
            return err;
        }

        // Step 1: Compute S = Q * K^T
        const int TILE_SIZE = 16;
        dim3 block_S(TILE_SIZE, TILE_SIZE);
        dim3 grid_S((seq_len + TILE_SIZE - 1) / TILE_SIZE,
                   (seq_len + TILE_SIZE - 1) / TILE_SIZE);

        compute_S<float, TILE_SIZE><<<grid_S, block_S>>>(
            Q_batch, K_tr_dev, seq_len, head_dim, S_dev
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(S_dev);
            cudaFree(P_dev);
            cudaFree(K_tr_dev);
            return err;
        }

        // Step 2: Compute P = softmax(S)
        // Use the shared memory version for better performance
        dim3 block_P(256);  // threads per block
        dim3 grid_P(seq_len);  // one block per row

        compute_P_shared_mem<float, TILE_SIZE><<<grid_P, block_P>>>(
            S_dev, P_dev, seq_len
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(S_dev);
            cudaFree(P_dev);
            cudaFree(K_tr_dev);
            return err;
        }

        // Step 3: Compute O = P * V
        dim3 block_O(TILE_SIZE, TILE_SIZE);
        dim3 grid_O((head_dim + TILE_SIZE - 1) / TILE_SIZE,
                   (seq_len + TILE_SIZE - 1) / TILE_SIZE);

        compute_O<float, TILE_SIZE><<<grid_O, block_O>>>(
            V_batch, P_dev, seq_len, head_dim, O_batch
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(S_dev);
            cudaFree(P_dev);
            cudaFree(K_tr_dev);
            return err;
        }

        // Cleanup temporary memory
        cudaFree(S_dev);
        cudaFree(P_dev);
        cudaFree(K_tr_dev);
    }

    return cudaSuccess;
}

#include <cuda_runtime.h>
#include <stdio.h>


// Forward declarations of the existing flash attention kernels
template<class ElTp, int T>
__global__ void flash_attention(ElTp* Q, ElTp* K, ElTp* V, ElTp* O, int N, int d,
                                ElTp* l, ElTp* m,
                               int T_c, int T_r, int B_c, int B_r);

// CUDA kernel launcher that interfaces with the torch wrapper
cudaError_t launch_flash_attention_kernels(
    float* Q_ptr, float* K_ptr, float* V_ptr, float* O_ptr,
    int seq_len, int head_dim
) {
    // TODO

    //get CUDA max shared memory per block
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t maxSharedMemPerBlock = prop.sharedMemPerBlock;

    // launch flash attention kernel
    int M = maxSharedMemPerBlock / sizeof(float); // max elements in shared memory
    // ceil(M/(4*head_dim))
    int B_c = (M + 4*head_dim - 1) / (4*head_dim);
    int B_r = std::min(B_c, head_dim);

    int T_c = (seq_len + B_c - 1) / B_c;
    int T_r = (seq_len + B_r - 1) / B_r;

    dim3 blockDim(32, 32); // 1024 threads
    dim3 gridDim(1, 1); // single block for now
    size_t sharedMemSize = (B_c * head_dim + B_c * head_dim + B_r * head_dim + B_r * B_c) * sizeof(float);


    if (sharedMemSize > maxSharedMemPerBlock) {
        printf("Error: Shared memory size %zu exceeds maximum %zu\n", sharedMemSize, maxSharedMemPerBlock);
        return cudaErrorInvalidValue;
    }

    flash_attention<float, 32><<<gridDim, blockDim, sharedMemSize>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, seq_len, head_dim,
        nullptr, nullptr, // TODO
        T_c, T_r, B_c, B_r
    );




    return cudaSuccess;
}

/*
 * Original flash attention kernel implementation
 * Q,K,V: (N, d)
 * N: sequence length
 * d: head dimension
 * M: size of SRAM, d<=M<=Nd
 */

/*
 * Hardware used for testing:
 * NVIDIA A100-PCIE-40GB
 * Shared memory per block: 48 KiB
 * Opt-in max shared memory per block: 163 KiB (may reduce occupancy)
 */

// non-causal flash attention kernel
template<class ElTp, int T>
__global__ void flash_attention(ElTp* Q, ElTp* K, ElTp* V, ElTp* O, int N, int d,
                                ElTp* l, ElTp* m,
                               int T_c, int T_r, int B_c, int B_r) {

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    // we have a 32*32 thread block

    //TODO: ceil
    const int tiles_x = d / blockDim.x;
    const int tiles_y = B_c / blockDim.y;

    // external so we can manage size outside kernel
    // size = (B_c*d + B_c*d + B_r*d + B_r*B_c)*sizeof(ElTp)
    // maxes out SRAM
    extern __shared__ float shared[];
    // assume O is initialized to zero
    ElTp* K_j = shared;         // [B_c][d]
    ElTp* V_j = &K_j[B_c*d];    // [B_c][d]
    ElTp* Q_i = &V_j[B_c*d];    // [B_r][d], make sure S_ij also fits
    ElTp* S_ij = &Q_i[B_r*d];   // [B_r][B_c]

    for (int j=0; j<T_c; j++){
        // load K_j, V_j

        // each thread loads #tiles elements
        for (int l_x=0; l_x<tiles_x; l_x++){
            for (int l_y=0; l_y<tiles_y; l_y++){
                int col = tid_x + l_x*blockDim.x;
                int row = tid_y + l_y*blockDim.y;
                if (col < d && row < B_c){
                    K_j[row*d + col] = K[(j*B_c + row)*d + col];
                    V_j[row*d + col] = V[(j*B_c + row)*d + col];
                }
            }
        }

        // ensure K_j, V_j are loaded
        __syncthreads();

        for (int i=0; i<T_r; i++){
            // load Q_i
            for (int l_x=0; l_x<tiles_x; l_x++){
                for (int l_y=0; l_y<tiles_y; l_y++){
                    int col = tid_x + l_x*blockDim.x;
                    int row = tid_y + l_y*blockDim.y;
                    if (col < d && row < B_r){
                        Q_i[row*d + col] = Q[(i*B_r + row)*d + col];
                    }
                }
            }
            // load l_i, m_i
            // these fit into registers
            ElTp l_i = l[i*B_r + tid_y];
            ElTp m_i = m[i*B_r + tid_y];
            // load O_i
        }
    }
}

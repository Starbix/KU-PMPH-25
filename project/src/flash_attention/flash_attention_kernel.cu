#include <cuda_runtime.h>
#include <stdio.h>


#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// Forward declarations of the existing flash attention kernels
template<class ElTp, int T>
__global__ void flash_attention(ElTp* Q, ElTp* K, ElTp* V, ElTp* O, int N, int d,
                                ElTp* l, ElTp* m,
                               int T_c, int T_r, int B_c, int B_r);



// CUDA kernel launcher that interfaces with the torch wrapper
template<class ElTp>
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

    // initialize l and m
    ElTp* l;
    ElTp* m;
    cudaMalloc(&l, seq_len * sizeof(ElTp));
    cudaMalloc(&m, seq_len * sizeof(ElTp));
    int B = 256,  num_blocks = CEIL_DIV(seq_len, B);
    dim3 block(B, 1, 1), grid(num_blocks, 1, 1);
    init_l<<<grid, block>>>(l, seq_len);
    init_m<<<grid, block>>>(m, seq_len);

    flash_attention<float, 32><<<gridDim, blockDim, sharedMemSize>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, seq_len, head_dim,
        l, m, 
        T_c, T_r, B_c, B_r
    );




    return cudaSuccess;
}


template<class ElTp>
__global__ void init_l(ElTp* l, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        l[idx] = 0.f;
    }
}

template<class ElTp>
__global__ void init_m(ElTp* m, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        m[idx] = -INFINITY;
    }
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
    extern __shared__ ElTp shared[];
    ElTp* Q_i = shared;
    ElTp* K_j = Q_i + B_r * d;
    ElTp* V_j = K_j + B_c * d;
    ElTp* S_ij = V_j + B_c * d;
    ElTp* O_i = S_ij + B_r * B_c;
    ElTp* l_i = O_i + B_r * d;
    ElTp* l_i_new = l_i + B_r;
    ElTp* m_i = l_i_new + B_r;
    ElTp* m_i_new = m_i + B_r;
    ElTp* m_ij_dash = m_i_new + B_r;

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

#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>


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
    printf("B_c: %d, B_r: %d\n", B_c, B_r);

    int T_c = (seq_len + B_c - 1) / B_c;
    int T_r = (seq_len + B_r - 1) / B_r;

    dim3 blockDim(32, 8); // 1024 threads
    dim3 gridDim(T_r, 1); //
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
// - batch size 1
// - single head
// - no masking
// - blockDim(B_r, B_c) TOO BIG
// - blockDim(32, 24) for A100 -> three elements per thread
// - blockDim(16, 16) -> 3x3 elements per thread
// - blockDim(32, 8) -> 256 threads,
//          makes loading nice and coalesced throughout 16 instead of 8 should also work well
// - B_r,B_c = 48 for A100 with max shared memory without opt-in
// - gridDim(1, ceil(N/B_r))
//
// biggest problem: shared mem arrays are larger than 1024 elements
// thus we can't simply rely on threadIds and need to do more work per thread
template<class ElTp, int T>
__global__ void flash_attention(ElTp* Q, ElTp* K, ElTp* V, ElTp* O, int N, int d,
                                ElTp* l, ElTp* m,
                               int T_c, int T_r, int B_c, int B_r) {

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    // external so we can manage size outside kernel
    // size = (B_c*d + B_c*d + B_r*d + B_r*B_c)*sizeof(ElTp)
    // maxes out SRAM
    extern __shared__ float shared[];
    // assume O is initialized to zero
    ElTp* K_j = shared;         // [B_c][d]
    ElTp* V_j = &K_j[B_c*d];    // [B_c][d]
    ElTp* Q_i = &V_j[B_c*d];    // [B_r][d], make sure S_ij also fits
    ElTp* S_ij = &Q_i[B_r*d];   // [B_r][B_c]
    // ElTp* O_i = &S_ij[B_r*B_c];   // [B_r][d], should fit in registers

    //TODO: hack for now
    const int num_tiles_x = 64/32;
    const int num_tiles_y = 48/8;
    // const int num_tiles_x = d/blockDim.y;
    // const int num_tiles_y = B_r/blockDim.y;
    // const int num_tiles = num_tiles_x * num_tiles_y;

    // each thread computes multiple output elements
    ElTp O_i[num_tiles_y][num_tiles_x]; // for 32*8 this should be 12 entries and fit into registers
    ElTp m_i[num_tiles_y]; // per row max
    ElTp m_last[num_tiles_y];
    ElTp l_i[num_tiles_y]; // per row sum

    // initialize O_i to zero
    #pragma unroll
    for (int y=0; y<num_tiles_y; y++){
        #pragma unroll
        for (int x=0; x<num_tiles_x; x++){
            O_i[y][x] = 0.f;
        }
        m_i[y] = -INFINITY;
        l_i[y] = 0.f;
    }


    // the outer T_r loop is done by the gridDim.x

    //load K_j and V_j
    // #pragma unroll
    // for (int row=tid_y; row<B_c; row+=blockDim.y){
    //     int global_row = row + blockIdx.y * B_c;
    //     if (global_row < N){
    //         #pragma unroll
    //         for (int col=tid_x;col<d; col+=blockDim.x){
    //             K_j[row*d + col] = K[global_row*d + col]; // coalesced cuz + tid_x
    //             V_j[row*d + col] = V[global_row*d + col];
    //         }

    //     }
    // }

    // load Q_i, rows of size B_r (48), but we only have 8 threads in y direction
    // in x direction we have 32 threads (coalesced), but still need to loop over d
    #pragma unroll
    for (int row=tid_y; row<B_r; row+=blockDim.y){
        int global_row = row + blockIdx.y * B_r;
        if (global_row < N){
            #pragma unroll
            for (int col=tid_x; col<d; col+=blockDim.x){
                Q_i[row*d + col] = Q[global_row*d + col]; // coalesced cuz + tid_x
            }
        }
    }
    // no need to load O_i as we initialize to zero in kernel,
    // because we iterate over i=0..T_r and each block handles one i block

    __syncthreads();

    for (int j=0; j<T_c; j++){

        // load K_j and V_j
        #pragma unroll
        for (int row=tid_y; row<B_c; row+=blockDim.y){
            int global_row = row + j * B_c;
            if (global_row < N){
                #pragma unroll
                for (int col=tid_x;col<d; col+=blockDim.x){
                    K_j[row*d + col] = K[global_row*d + col]; // coalesced cuz + tid_x
                    V_j[row*d + col] = V[global_row*d + col];
                }
            }
        }

        // compute S_ij = Q_i K_j^T
        // possibly move compuation inside above loop to hide latency of mem loads
        __syncthreads();
        #pragma unroll
        for (int row=tid_y; row<B_r; row+=blockDim.y){
            #pragma unroll
            for (int col=tid_x; col<B_c; col+=blockDim.x){
                ElTp S_ij_val = 0.f;
                #pragma unroll
                for (int t=0; t<d; t++){
                    S_ij_val += Q_i[row*d + t] * K_j[col*d + t]; // K_j is stored row major but we need K_j^T
                }
                S_ij[row*B_c + col] = S_ij_val;
            }
        }

        // compute row-wise max m_i
        // we don't have enough threads in the y dimension as 8<B_r(48)
        // this means we need to remap the threads we do have (we assume total threads >= B_r)
        // we don't need to care too much about which thread does what as long as all rows are covered
        // because we are working with shared memory here, so no coalescing issues
        __syncthreads();
        #pragma unroll
        for (int row=tid_y; row<B_r; row+=blockDim.y){
            m_last[row] = m_i[row]; // store last max
            ElTp m = m_i[row];
            #pragma unroll
            for (int col=0; col<B_c; col++){
                ElTp val = S_ij[row*B_c + col];
                if (m < val){
                    m = val;
                }
            }
            m_i[row] = m;
        }
        // renormalize O_i: e^(m_last - m_i)
        if (j>0){
            #pragma unroll
            for (int y=0; y<num_tiles_y; y++){
                ElTp m_diff = exp(m_last[y] - m_i[y]);
                #pragma unroll
                for (int x=0; x<num_tiles_x; x++){
                    O_i[y][x] *= m_diff;
                }
            }
        }

        __syncthreads();
        // threads 0..48 now have m_i

    }

    // here we write back
    // original thread indices cause of coalescing

}

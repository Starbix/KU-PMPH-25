/*
Some of the code in this file is inspired by the lecture notes on how to make
tiled Matrix Multiplication, chapter 6.3.
*/

#include <cmath>
template<class ElTp, int T>
__global__ void compute_S(ElTp* Q, ElTp* K_tr, int N, int d, ElTp* S) {

    __shared__ ElTp Q_block[T][T];
    __shared__ ElTp K_tr_block[T][T];

    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;

    ElTp s = 0.0;
    for (int kk = 0; kk < d; kk += T) {
        // load Q and K^T by blocks from global memory
        Q_block[threadIdx.y][threadIdx.x] = (row < N && kk + threadIdx.x < d) ?
                                            Q[row*d + kk + threadIdx.x] : 0.0;
        K_tr_block[threadIdx.y][threadIdx.x] = (col < N && kk + threadIdx.y < d) ?
                                               K_tr[(kk + threadIdx.y)*N + col] : 0.0;

        __syncthreads();

        // compute S = Q_block * K_block
        #pragma unroll
        for (int k = 0; k < T; k++) {
            s += Q_block[threadIdx.y][k] * K_tr_block[k][threadIdx.x];
        }

        __syncthreads();
    }

    // write S to global memory
    if (row < N && col < N)
        S[row*N+ col] = s;
}

// naive kernel
// non-coalesced access to global memory
template<class ElTp>
__global__ void compute_P_naive(ElTp* S, ElTp* P, int N) {
    int row = blockDim.x*blockIdx.x + threadIdx.x;

    if (row<N) {
        ElTp max_el = -INFINITY;
        ElTp norm = 0;

        //calculate max_el for better numerical stability
        for (int i=0; i<N; i++){
            max_el = max(max_el, S[row*N + i]);
        }

        //calculate norm
        for (int i=0; i<N; i++){
            norm += exp(S[row*N + i] - max_el);
        }

        //write to P
        for (int i=0; i<N; i++){
            P[row*N + i] = exp(S[row*N + i] - max_el)/norm;
        }
    }
}

template<class ElTp>
__global__ void compute_P_online(ElTp* S, ElTp* P, int N) {
    int row = blockDim.x*blockIdx.x + threadIdx.x;

    if (row<N) {
        ElTp max_el = -INFINITY;
        ElTp norm = 0;

        //single pass for max computation and norm
        for (int i=0; i<N; i++){
            ElTp curr_el = S[row*N + i];

            if (curr_el>max_el){
                norm *= exp(max_el - curr_el);
                max_el = curr_el;
            }
            norm += exp(curr_el - max_el);
        }

        //write to P
        for (int i=0; i<N; i++){
            P[row*N + i] = exp(S[row*N + i] - max_el)/norm;
        }
    }
}


// this kernel makes use of shared memory and reductions
// it also ensures coalesced memory access
// each block computes one row
template<class ElTp, int T>
__global__ void compute_P_shared_mem(ElTp* S, ElTp* P, int N) {

    // shared memory
    // TODO: make it more flexible
    __shared__ ElTp scratch[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row>=N) return;

    ElTp thread_max = -INFINITY;
    ElTp thread_norm = 0;

    #pragma unroll
    for (int i=tid; i<N; i+=blockDim.x){
        // access in coalesced fashion
        ElTp el = S[row*N + i];

        if (el>thread_max){
            thread_norm *= exp(thread_max - el);
            thread_max = el;
        }

        thread_norm += exp(el - thread_max);
    }

    scratch[tid] = thread_max;
    // make sure all threads on block have computed their norm and max
    __syncthreads();

    // reduce max over block
    // example with blockDim.x=1024
    // tid=0 will first compute max of tid=0 and tid=512 and write it into scratch[0]
    // then it'll combine its result with the one from tid=256, so max of (0, 256, 768, 1024)
    for (int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s){
            scratch[tid] = max(scratch[tid], scratch[tid + s]);
        }
        __syncthreads();
    }
    // the block-global has been reduced in tid=0
    ElTp row_max = scratch[0];
    __syncthreads();

    // write local norm to scratch
    scratch[tid] = thread_norm * exp(thread_max - row_max);
    __syncthreads();

    // reduce norm over threads
    for (int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s){
            scratch[tid] += scratch[tid + s];
        }
        __syncthreads();
    }
    // the block-global has been reduced in tid=0
    ElTp row_norm = scratch[0];
    __syncthreads();

    #pragma unroll
    for (int i=tid; i<N; i+=blockDim.x){
        P[row*N + i] = exp(S[row*N + i] - row_max)/row_norm;
    }
}


template<class ElTp, int T>
__global__ void compute_O(ElTp* V, ElTp* P, int N, int d, ElTp* O) {

    __shared__ ElTp P_block[T][T];
    __shared__ ElTp V_block[T][T];

    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;

    ElTp o = 0.0;
    for (int kk = 0; kk < N; kk += T) {
        // load P and V by blocks from global memory
        P_block[threadIdx.y][threadIdx.x] = (row < N && kk + threadIdx.x < N) ?
                                            P[row*N + kk + threadIdx.x] : 0.0;
        V_block[threadIdx.y][threadIdx.x] = (col < d && kk + threadIdx.y < N) ?
                                               V[(kk + threadIdx.y)*d + col] : 0.0;

        __syncthreads();

        // compute O = P * V
        #pragma unroll
        for (int k = 0; k < T; k++) {
            o += P_block[threadIdx.y][k] * V_block[k][threadIdx.x];
        }

        __syncthreads();
    }

    // write O to global memory
    if (row < N && col < d)
        O[row*d + col] = o;
}

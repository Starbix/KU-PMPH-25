


template<class ElTp, int T>
__global__ void attention_kernel(ElTp* Q, ElTp* K_tr, ElTp* V, ElTp* S, ElTp* P, int N, int d, ElTp* O) {
    
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

    // read S from global memory 

    // compute P = softmax(S)

    // write P to global memory

    __shared__ ElTp P_block[T][T];
    __shared__ ElTp V_block[T][T];
        
    ElTp o = 0.0;
    for (int kk = 0; kk < d; kk += T) {
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

    // done
}
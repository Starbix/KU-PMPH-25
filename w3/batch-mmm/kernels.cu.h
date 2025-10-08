#ifndef BMMM_KERNELS
#define BMMM_KERNELS

/**
 * We use ElTp for the (generic) array-element type.
 * For simplicity, this kernel assumes a grid of M blocks
 * (on the x dimension) and that the block size on the x and
 * y dimensions is K, i.e., blockDim.x == blockDim.y == K.
 */

template <class ElTp> __global__
void bmmmNaiveKer ( ElTp* A, ElTp* B, char* X, ElTp* Y
                  , const int M,  const int K, const int N
) {
  const int i  = blockIdx.x;
  const int j1 = threadIdx.y;
  const int j2 = threadIdx.x;

  // sanity checks
  if( (i >= M) || (j1 >= K) )
    return;

  ElTp acc = 0.0f;

  for(int q=0; q<N; q++) { // reduction
    float v = 0.0;
    if( X[i*N + q] != 0 ) {
      v = 1.0;
    }
    ElTp a = A[j1*N + q];
    ElTp b = B[q*K + j2];
    acc += a*b*v;
  }
  Y[i*K*K + j1*K + j2] = acc;
}


/**
 * This is not the optimal version as it
 *   does not efficiently sequentializes.
 * Assumes:
 *    blockDim.y == blockDim.x == T
 */
template <class ElTp, int T>
__global__ void matTransposeTiledKer(ElTp* A, ElTp* A_tr, const int heightA, const int widthA) {
  __shared__ ElTp tile[T][T+1];

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;

  if( x < widthA && y < heightA )
      tile[threadIdx.y][threadIdx.x] = A[y*widthA + x];

  __syncthreads();

  x = blockIdx.y * T + threadIdx.x;
  y = blockIdx.x * T + threadIdx.y;

  if( x < heightA && y < widthA )
      A_tr[y*heightA + x] = tile[threadIdx.x][threadIdx.y];
}


/**
 * ElTp is some numeric type, e.g., float.
 * T is the size of the tile used for the
 *    outermost loop of count M.
 *
 * Array dimensions are as in goldenSeq;
 * X_tr is the transposed on X.
 * A:[K][N] , B:[N][K] , X_tr:[N][M] , Y:[M][K][K]
 * Assumes:
 *    (1) blockDim.y * blockDim.x >= T
 *    (2) blockDim.x == blockDim.y == K
 */
template <class ElTp, int T> __global__
void bmmmTiledKer ( ElTp* A,      ElTp* B, char* X_tr,   ElTp* Y
                  , const int M,  const int K, const int N
) {
  __shared__ ElTp Xsh_tr[T];
  ElTp acc[T];

  const int ii  = blockIdx.x * T;
  const int j1  = threadIdx.y;
  const int j2  = threadIdx.x;
  const int flat_thid = threadIdx.y * K + threadIdx.x;

  #pragma unroll
  for(int t=0; t<T; t++)
    acc[t] = 0;

  /***********************************************
   *** Cuda Exercise 4: ***
   *
   * With the help of the pseudocode from the
   * lecture slides, please implement the rest of
   * the code of this kernel.
   * Remember to flatten the indices to all arrays
   * hold in global memory, i.e., A, B, X_tr, Y.
   ***********************************************/

   for (int q=0; q<N; q++){
       ElTp ab = A[j1*N + q] * B[q*K + j2];

       int i = ii + flat_thid;

       char x = (flat_thid<T && i<M) ? X_tr[q*M + i] : 0;
       Xsh_tr[flat_thid] = x;

       // Xsh_tr needs to be fully written before being read
       __syncthreads();

       #pragma unroll
       for (int i_r=0; i_r<T; i_r++){
           ElTp v = (Xsh_tr[i_r]!=0) ? 1 : 0;
           acc[i_r] += ab*v;
       }
       // Xsh_tr will be overwritten in the next iteration, so make sure all threads are done using it
       __syncthreads();
   }

   for (int i_r=0; i_r<T; i_r++){
       if (ii+i_r<M){
           // Y:[M][K][K]
           // Y[ii+i_r][j1][j2]
           Y[(ii+i_r)*K*K + j1*K + j2] = acc[i_r];
       }
   }

}
#endif

#include "../utils.h"
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define DEBUG 0

#define WARP_LEVEL_REDUCE 0

#if WARP_LEVEL_REDUCE
template <typename ElTp>
__device__ __forceinline__ ElTp warpReduceMax(ElTp val) {
// Assuming warp size is 32
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    ElTp other = __shfl_down_sync(0xffffffff, val, offset);
    val = max(val, other);
  }
  return val;
}
#endif

template <typename ElTp, int B_c, int B_r, int d, int bdim_x, int bdim_y>
__global__ void flash_attention(ElTp *Q, ElTp *K, ElTp *V, ElTp *O, int N,
                                int T_c);

#define INSTANTIATE_KERNEL(HEAD_DIM, BC, BR, BDIM_X, BDIM_Y)                   \
  if (head_dim == HEAD_DIM && B_c == BC && B_r == BR && bdim_x == BDIM_X &&    \
      bdim_y == BDIM_Y) {                                                      \
    flash_attention<float, BC, BR, HEAD_DIM, BDIM_X, BDIM_Y>                   \
        <<<gridDim, blockDim, sharedMemSize>>>(Q_ptr, K_ptr, V_ptr, O_ptr,     \
                                               seq_len, T_c);                  \
    return cudaSuccess;                                                        \
  }

// This function can be used for parameter optimization
cudaError_t launch_flash_attention_kernels_with_params(
    float *Q_ptr, float *K_ptr, float *V_ptr, float *O_ptr, int seq_len,
    int head_dim, int B_c, int B_r, int bdim_x, int bdim_y) {
  // get CUDA max shared memory per block
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  size_t maxSharedMemPerBlock = prop.sharedMemPerBlock;

  int T_c = CEIL_DIV(seq_len, B_c);
  int T_r = CEIL_DIV(seq_len, B_r);

  dim3 blockDim(bdim_x, bdim_y);

  if (DEBUG) {
    printf("blockDim: (%d, %d)\n", blockDim.x, blockDim.y);
    printf("T_c: %d, T_r: %d\n", T_c, T_r);
    printf("B_c: %d, B_r: %d\n", B_c, B_r);
    printf("head_dim: %d\n", head_dim);
  }
  dim3 gridDim(T_r, 1);

  // K_j, V_j, Q_i, S_ij
  size_t sharedMemSize =
      (2*B_c * head_dim + B_r * head_dim + B_r * B_c) *
      sizeof(float);

  if (sharedMemSize > maxSharedMemPerBlock) {
      printf("Error: Shared memory size %zu exceeds maximum %zu\n",
      sharedMemSize, maxSharedMemPerBlock);
      return cudaErrorMemoryAllocation;
  }
  // fastest configs
  INSTANTIATE_KERNEL(64, 32, 16, 32, 16)
  INSTANTIATE_KERNEL(128, 32, 16, 32, 16)

  // if you want to try out different parameter configurations, 
  // you should add INSTANTIATE_KERNEL with them here. For example:
  // INSTANTIATE_KERNEL(128, 48, 32, 16, 16)

  printf("Error: Unsupported configuration - head_dim=%d, B_c=%d, B_r=%d, "
         "bdim_x=%d, bdim_y=%d\n",
         head_dim, B_c, B_r, bdim_x, bdim_y);
  return cudaErrorInvalidConfiguration;
}

cudaError_t launch_flash_attention_kernels(float *Q_ptr, float *K_ptr,
                                           float *V_ptr, float *O_ptr,
                                           int seq_len, int head_dim) {
  // get CUDA max shared memory per block
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  size_t maxSharedMemPerBlock = prop.sharedMemPerBlock;

  // launch flash attention kernel
  int M = maxSharedMemPerBlock / sizeof(float); // max elements in shared memory

  // like in paper
  // int B_c = std::min(CEIL_DIV(M, 4 * head_dim), seq_len);
  // int B_r = std::min(B_c, head_dim);
  const int B_c = 32;
  const int B_r = 16;

  // static
  int bdim_x = 32;
  int bdim_y = 16;

  return launch_flash_attention_kernels_with_params(
      Q_ptr, K_ptr, V_ptr, O_ptr, seq_len, head_dim, B_c, B_r, bdim_x, bdim_y);
}

#undef INSTANTIATE_KERNEL

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
//          makes loading nice and coalesced throughout 16 instead of 8 should
//          also work well
// - B_r,B_c = 48 for A100 with max shared memory without opt-in
// - gridDim(1, ceil(N/B_r))
//
// biggest problem: shared mem arrays are larger than 1024 elements
// thus we can't simply rely on threadIds and need to do more work per thread
template <class ElTp, int B_c, int B_r, const int d, const int bdim_x,
          const int bdim_y>
__global__ void flash_attention(ElTp *Q, ElTp *K, ElTp *V, ElTp *O, int N,
                                int T_c) {

  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;

  // external so we can manage size outside kernel

  extern __shared__ float shared[];
  // assume O is initialized to zero
  ElTp *K_j = shared;         // [B_c][d]
  ElTp *V_j = &K_j[B_c * d];  // [B_c][d]
  ElTp *Q_i = &V_j[B_c * d];  // [B_r][d], make sure S_ij also fits
  ElTp *S_ij = &Q_i[B_r * d]; // [B_r][B_c]

  const int num_tiles_x = CEIL_DIV(d, bdim_x);
  const int num_tiles_y = CEIL_DIV(B_r, bdim_y);

  // each thread computes multiple output elements
  // this is thread-local memory
  // they should be small enough to fit in registers
  // if they don't fit, they will spill to local memory
  // which is basically global memory and very slow
  // AVOID THAT
  // if we were using CUDA 13, we could spill to shared memory instead
  ElTp O_i[num_tiles_y][num_tiles_x];
  ElTp m_i[num_tiles_y]; // per row max
  ElTp m_last[num_tiles_y];
  ElTp l_i[num_tiles_y]; // per row sum

// initialize O_i to zero
#pragma unroll
  for (int y = 0; y < num_tiles_y; y++) {
#pragma unroll
    for (int x = 0; x < num_tiles_x; x++) {
      O_i[y][x] = 0.f;
    }
    m_i[y] = -INFINITY;
    l_i[y] = 0.f;
  }

// the outer T_r loop is done by the gridDim.x

// load Q_i, rows of size B_r (48), but we only have 8 threads in y direction
// in x direction we have 32 threads (coalesced), but still need to loop over d
#pragma unroll
  for (int row = tid_y; row < B_r; row += bdim_y) {
    int global_row = row + blockIdx.x * B_r;
    if (global_row < N) {
#pragma unroll
      for (int col = tid_x; col < d; col += bdim_x) {
        Q_i[row * d + col] = Q[global_row * d + col]; // coalesced cuz + tid_x
      }
      // if B_r doesn't divide N, need to zero out extra rows
    } else {
#pragma unroll
      for (int col = tid_x; col < d; col += bdim_x) {
        Q_i[row * d + col] = 0.f;
      }
    }
  }
  // no need to load O_i as we initialize to zero in kernel,
  // because we iterate over i=0..T_r and each block handles one i block

  __syncthreads();

  for (int j = 0; j < T_c; j++) {
    if (DEBUG && blockIdx.x == 0 && tid_x == 0 && tid_y == 0)
      printf("j=%d\n", j);

// load K_j and V_j
#pragma unroll
    for (int row = tid_y; row < B_c; row += bdim_y) {
      int global_row = row + j * B_c;
      if (global_row < N) {
#pragma unroll
        for (int col = tid_x; col < d; col += bdim_x) {
          // TODO: handle non-divisible d
          K_j[row * d + col] = K[global_row * d + col]; // coalesced cuz + tid_x
          V_j[row * d + col] = V[global_row * d + col];
        }
        // if B_c doesn't divide N, need to zero out extra rows
      } else {
#pragma unroll
        for (int col = tid_x; col < d; col += bdim_x) {
          K_j[row * d + col] = 0.f;
          V_j[row * d + col] = 0.f;
        }
      }
    }
    __syncthreads();
    // debug print K_j
    if (0 && DEBUG && blockIdx.x == 0 && blockIdx.y == 0 && tid_x == 0 &&
        tid_y == 0) {
      printf("Thread (block=%d,%d; thread=%d,%d) printing for j=%d\n",
             blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, j);
      printf("K_j (j=%d):\n", j);
      for (int row = 0; row < B_c; row++) {
        printf("row: %d: ", row);
        for (int col = 0; col < d; col++) {
          printf("%.0f ", K_j[row * d + col]);
        }
        printf("\n");
      }
    }

// compute S_ij = Q_i K_j^T
// possibly move compuation inside above loop to hide latency of mem loads
#pragma unroll
    for (int row = tid_y; row < B_r; row += bdim_y) {
#pragma unroll
      for (int col = tid_x; col < B_c; col += bdim_x) {
        if (0 & DEBUG && tid_y == 0 && tid_x == 0) {
          printf("Computing S_ij for row %d, col %d\n", row, col);
        }
        ElTp S_ij_val = 0.f;
#pragma unroll
        for (int t = 0; t < d; t++) {
          S_ij_val +=
              Q_i[row * d + t] *
              K_j[col * d + t]; // K_j is stored row major but we need K_j^T
        }
        S_ij[row * B_c + col] = S_ij_val;
      }
    }
    if (DEBUG)
      __syncthreads();

    if (0 && DEBUG && blockIdx.x == 0 && blockIdx.y == 0 && tid_x == 0 &&
        tid_y == 0) {
      printf("S_ij (j=%d):\n", j);
      printf("row: %d: ", 0);
      for (int col = 0; col < B_c; col++) {
        printf("%.0f ", S_ij[col]);
      }
      printf("\n");
    }

// compute row-wise max m_i
// we don't have enough threads in the y dimension as 8<B_r(48)
// this means we need to remap the threads we do have (we assume total threads
// >= B_r) we don't need to care too much about which thread does what as long
// as all rows are covered because we are working with shared memory here, so no
// coalescing issues
#if WARP_LEVEL_REDUCE
// warp-level reduction for parallel max computation
// DANGEROUS: only if bdim_x==32
#pragma unroll
    for (int row = tid_y; row < B_r; row += bdim_y) {
      int reg_row = row / bdim_y; // which row in registers

      m_last[reg_row] = m_i[reg_row]; // store last max

      // Each thread computes partial max over its columns
      ElTp thread_max = -INFINITY;
#pragma unroll
      for (int col = tid_x; col < B_c; col += bdim_x) {
        ElTp val = S_ij[row * B_c + col];
        thread_max = max(thread_max, val);
      }

      ElTp warp_max = warpReduceMax(thread_max);

      // broadcast result to all threads in the warp
      warp_max = __shfl_sync(0xffffffff, warp_max, 0);

      // Update with previous max
      m_i[reg_row] = max(m_i[reg_row], warp_max);
    }
#else
#pragma unroll
    for (int row = tid_y; row < B_r; row += bdim_y) {
      int reg_row = row / bdim_y; // which row in registers
      if (0 && DEBUG && tid_y == 0 && tid_x == 0) {
        printf("Computing m_i for row %d and reg_row %d\n", row, reg_row);
      }
      m_last[reg_row] = m_i[reg_row]; // store last max
      ElTp m = m_i[reg_row];
#pragma unroll
      for (int col = 0; col < B_c; col++) {
        ElTp val = S_ij[row * B_c + col];
        if (m < val) {
          m = val;
        }
      }
      if (0 && DEBUG && tid_y == 0 && tid_x == 0) {
        printf("Computed m_i for row %d: %.0f (last: %.0f)\n", row, m,
               m_last[reg_row]);
      }
      m_i[reg_row] = m;
    }
#endif

    if (0 && DEBUG && blockIdx.x == 0 && blockIdx.y == 0 && tid_x == 0 &&
        tid_y == 0) {
      printf("m_i (j=%d): ", j);
      for (int row = 0; row < num_tiles_y; row++) {
        printf("%.0f ", m_i[row]);
      }
      printf("\n");
      printf("l_i (j=%d): ", j);
      for (int row = 0; row < num_tiles_y; row++) {
        printf("%.0f ", l_i[row]);
      }
      printf("\n");
    }

    // renormalize O_i: e^(m_last - m_i)
    if (j > 0) {
#pragma unroll
      for (int y = 0; y < num_tiles_y; y++) {
        ElTp m_diff = exp(m_last[y] - m_i[y]);
#pragma unroll
        for (int x = 0; x < num_tiles_x; x++) {
          O_i[y][x] *= m_diff;
        }
        // also renormalize l_i
        l_i[y] *= m_diff;
      }
    }
// compute l_i and O_i
#pragma unroll
    for (int row = tid_y; row < B_r; row += bdim_y) {
      const int reg_row = row / bdim_y;
      ElTp l_val = 0.f;
#pragma unroll
      for (int col = 0; col < B_c; col++) {
        ElTp S_ij_exp = exp(S_ij[row * B_c + col] - m_i[reg_row]);
        l_val += S_ij_exp;
// accumulate O_i
#pragma unroll
        for (int t = 0; t < num_tiles_x; t++) {
          int col_o = t * bdim_x + tid_x;
          if (col_o < d) {
            O_i[reg_row][t] += S_ij_exp * V_j[col * d + col_o];
          }
        }
      }
      
      l_i[reg_row] += l_val;
    }
   
    // threads 0..48 now have m_i
    //
    // debug print l_i
    if (0 && DEBUG && blockIdx.x == 0 && blockIdx.y == 0 && tid_x == 0 &&
        tid_y == 0) {
      printf("l_i after j=%d: ", j);
      for (int row = 0; row < num_tiles_y; row++) {
        printf("%.0f ", l_i[row]);
      }
      printf("\n");
    }
    // debug print O_i
    if (1 && DEBUG && blockIdx.x == 0 && blockIdx.y == 0 && tid_x == 0 &&
        tid_y == 0) {
      printf("O_i after j=%d:\n", j);
      for (int row = 0; row < num_tiles_y; row++) {
        printf("row %d: ", row);
        for (int col = 0; col < num_tiles_x; col++) {
          printf("%.0f ", O_i[row][col]);
        }
        printf("\n");
      }
    }
    __syncthreads();
  }

// here we write back
#pragma unroll
  for (int row = tid_y; row < B_r; row += bdim_y) {
    int global_row = row + blockIdx.x * B_r;
    if (global_row < N) {
// write O_i
#pragma unroll
      for (int col = tid_x; col < d; col += bdim_x) {
        // normalize with l_i
        const int reg_row = row / bdim_y;
        O[global_row * d + col] = O_i[reg_row][col / bdim_x] / l_i[reg_row];
      }
    }
  }
}

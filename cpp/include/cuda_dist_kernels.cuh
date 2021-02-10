// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include "cuda_base_kernels.cuh"


namespace cuhnsw {

// https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
__inline__ __device__
cuda_scalar warp_reduce_sum(cuda_scalar val) {
  #if __CUDACC_VER_MAJOR__ >= 9
  // __shfl_down is deprecated with cuda 9+. use newer variants
  unsigned int active = __activemask();
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      val = add(val, __shfl_down_sync(active, val, offset));
  }
  #else
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      val = add(val, __shfl_down(val, offset));
  }
  #endif
  return val;
}

__inline__ __device__
cuda_scalar dot(const cuda_scalar * a, const cuda_scalar * b, const int num_dims) {
  __syncthreads();
  static __shared__ cuda_scalar shared[32];

  // figure out the warp/ position inside the warp
  int warp =  threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // partially reduce the dot product inside each warp using a shuffle
  cuda_scalar val = 0;
  for (int i = threadIdx.x; i < num_dims; i += blockDim.x)
    val = add(val, mul(a[i], b[i]));
  val = warp_reduce_sum(val);
  
  // write out the partial reduction to shared memory if appropiate
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  // if we we don't have multiple warps, we're done
  if (blockDim.x <= WARP_SIZE) {
    return shared[0];
  }

  // otherwise reduce again in the first warp
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane]: conversion(0.0f);
  if (warp == 0) {
    val = warp_reduce_sum(val);
    // broadcast back to shared memory
    if (threadIdx.x == 0) {
        shared[0] = val;
    }
  }
  __syncthreads();
  return shared[0];
}

__inline__ __device__
cuda_scalar squaresum(const cuda_scalar * a, const cuda_scalar * b, const int num_dims) {
  __syncthreads();
  static __shared__ cuda_scalar shared[32];

  // figure out the warp/ position inside the warp
  int warp =  threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // partially reduce the dot product inside each warp using a shuffle
  cuda_scalar val = 0;
  for (int i = threadIdx.x; i < num_dims; i += blockDim.x) {
    cuda_scalar _val = sub(a[i], b[i]);
    val = add(val, mul(_val, _val));
  }
  __syncthreads();
  val = warp_reduce_sum(val);

  // write out the partial reduction to shared memory if appropiate
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  // if we we don't have multiple warps, we're done
  if (blockDim.x <= WARP_SIZE) {
    return shared[0];
  }

  // otherwise reduce again in the first warp
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane]: conversion(0.0f);
  if (warp == 0) {
    val = warp_reduce_sum(val);
    // broadcast back to shared memory
    if (threadIdx.x == 0) {
        shared[0] = val;
    }
  }
  __syncthreads();
  return shared[0];
}

__inline__ __device__
cuda_scalar GetDistanceByVec(const cuda_scalar* src_vec, const cuda_scalar* dst_vec, const int num_dims, const int dist_type) {
  cuda_scalar dist = 0;
  switch (dist_type) {
    case DOT:
      dist = -dot(src_vec, dst_vec, num_dims); break;
    case L2:
      dist = squaresum(src_vec, dst_vec, num_dims); break;
    default:
      break;
  }
  return dist;
}

__inline__ __device__
cuda_scalar GetDistance(const int srcid, const int dstid, const int num_dims,
    const int dist_type, const int* nodes, const cuda_scalar* data) {
  const cuda_scalar* src_vec = data + num_dims * nodes[srcid];
  const cuda_scalar* dst_vec = data + num_dims * nodes[dstid];
  return GetDistanceByVec(src_vec, dst_vec, num_dims, dist_type);
}

__inline__ __device__
cuda_scalar GetDistance2(const int src, const int dst, const int num_dims,
    const int dist_type, const cuda_scalar* data) {
  const cuda_scalar* src_vec = data + num_dims * src;
  const cuda_scalar* dst_vec = data + num_dims * dst;
  return GetDistanceByVec(src_vec, dst_vec, num_dims, dist_type);
}


__global__ void BatchDistanceKernel(
    const cuda_scalar* data, const int* src, const int* dst,
    const int size, const int num_dims, const int dist_type,
    float* distances) {
  for (int idx = blockIdx.x; idx < size; idx += gridDim.x) {
    const int _src = src[idx], _dst = dst[idx];
    cuda_scalar dist = GetDistance2(_src, _dst, num_dims, dist_type, data);
    #ifdef HALF_PRECISION
      if (threadIdx.x == 0) distances[idx] = __half2float(dist);
    #else
      if (threadIdx.x == 0) distances[idx] = dist;
    #endif
  }
}


} // namespace cuhnsw

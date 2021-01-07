// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include "cuda_base_kernels.cuh"
#include "cuda_dist_kernels.cuh"
#include "cuda_heap_kernels.cuh"


namespace cuhnsw {

__inline__ __device__
int warp_reduce_cand(const Neighbor* pq, int cand, const bool reverse) {
  #if __CUDACC_VER_MAJOR__ >= 9
  unsigned int active = __activemask();
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    int _cand = __shfl_down_sync(active, cand, offset);
    if (_cand >= 0) {
      if (cand == -1) {
        cand = _cand;
      } else {
        bool update = reverse? 
          lt(pq[cand].distance, pq[_cand].distance): 
          gt(pq[cand].distance, pq[_cand].distance);
        if (update) cand = _cand;
      }
    }
  }
  #else
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    int _cand = __shfl_down(cand, offset);
    if (_cand >= 0) {
      if (cand == -1) {
        cand = _cand;
      } else {
        bool update = reverse? 
          lt(pq[cand].distance, pq[_cand.distance]): 
          gt(pq[cand].distance, pq[_cand.distance]);
        if (update) cand = _cand;
      }
    }
  }
  #endif
  return cand;
}
__inline__ __device__
bool CheckAlreadyExists(const Neighbor* pq, const int size, const int nodeid) {
  __syncthreads();
  // figure out the warp/ position inside the warp
  int warp =  threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  
  static __shared__ bool shared[WARP_SIZE];
  bool exists = false;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    if (pq[i].nodeid == nodeid) {
      exists = true;
    }
  }
  
  #if __CUDACC_VER_MAJOR__ >= 9
  unsigned int active = __activemask();
  exists = __any_sync(active, exists);
  #else
  exists = __any(exists);
  #endif
  // write out the partial reduction to shared memory if appropiate
  if (lane == 0) {
    shared[warp] = exists;
  }
  
  __syncthreads();
  
  // if we we don't have multiple warps, we're done
  if (blockDim.x <= WARP_SIZE) {
    return shared[0];
  } 


  // otherwise reduce again in the first warp
  exists = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : false;
  if (warp == 0) {
    #if __CUDACC_VER_MAJOR__ >= 9
    active = __activemask();
    exists = __any_sync(active, exists);
    #else
    exists = __any(exists);
    #endif
    // broadcast back to shared memory
    if (threadIdx.x == 0) {
        shared[0] = exists;
    }
  }
  __syncthreads();
  return shared[0];



}
__inline__ __device__
int GetCand(const Neighbor* pq, const int size, const bool reverse) {
  __syncthreads();
  
  // figure out the warp/ position inside the warp
  int warp =  threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  static __shared__ int shared[WARP_SIZE];
  // pick the closest neighbor with checked = false if reverse = false and vice versa 
  cuda_scalar dist = reverse? -INFINITY: INFINITY;
  int cand = -1;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    if (not pq[i].checked) {
      bool update = reverse? lt(dist, pq[i].distance): gt(dist, pq[i].distance);
      if (update) {
        cand = i;
        dist = pq[i].distance;
      }
    }
  }
  cand = warp_reduce_cand(pq, cand, reverse);
  

  // write out the partial reduction to shared memory if appropiate
  if (lane == 0) {
    shared[warp] = cand;
  }
  __syncthreads();
  
  // if we we don't have multiple warps, we're done
  if (blockDim.x <= WARP_SIZE) {
    return shared[0];
  } 


  // otherwise reduce again in the first warp
  cand = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -1;
  if (warp == 0) {
    cand = warp_reduce_cand(pq, cand, reverse);
    // broadcast back to shared memory
    if (threadIdx.x == 0) {
        shared[0] = cand;
    }
  }
  __syncthreads();
  return shared[0];
}

__inline__ __device__
void PushNodeToPq(Neighbor* pq, int* size, const int max_size,
    const cuda_scalar* data, const int num_dims, const int dist_type,
    const int srcid, const int dstid, const int* nodes) {
  if (srcid == dstid) return;
  if (CheckAlreadyExists(pq, *size, dstid)) return;
  cuda_scalar dist = GetDistance(srcid, dstid, num_dims, dist_type, nodes, data);
  __syncthreads();
  if (*size < max_size) {
    PqPush(pq, size, dist, dstid, false);
  } else if (gt(pq[0].distance, dist)) {
    PqPop(pq, size);
    PqPush(pq, size, dist, dstid, false);
  }
  __syncthreads();
}

__inline__ __device__
void PushNodeToPq2(Neighbor* pq, int* size, const int max_size,
    const cuda_scalar dist, const int srcid, const int dstid, const int* nodes) {
  if (srcid == dstid) return;
  if (CheckAlreadyExists(pq, *size, dstid)) return;
  __syncthreads();
  if (*size < max_size) {
    PqPush(pq, size, dist, dstid, false);
  } else if (gt(pq[0].distance, dist)) {
    PqPop(pq, size);
    PqPush(pq, size, dist, dstid, false);
  }
  __syncthreads();
}

// similar to bloom filter
// while bloom filter prevents false negative, this visited table prevents false positive
// if it says the node is visited, it is actually visited
// if it says the node is not visited, it can be possibly visited
__inline__ __device__
bool CheckVisited(int* visited_table, int* visited_list, int& visited_cnt, int target, 
    const int visited_table_size, const int visited_list_size) {
  __syncthreads();
  bool ret = false;
  if (visited_cnt < visited_list_size ){
    int idx = target % visited_table_size;
    if (visited_table[idx] != target) {
      __syncthreads(); 
      if (threadIdx.x == 0) {
        if (visited_table[idx] == -1) {
          visited_table[idx] = target;
          visited_list[visited_cnt++] = idx;
        }
      }
    } else {
      ret = true;
    }
  }
  __syncthreads();
  return ret;
}

__inline__ __device__
void PushNodeToSearchPq(Neighbor* pq, int* size, const int max_size,
    const cuda_scalar* data, const int num_dims, const int dist_type,
    const cuda_scalar* src_vec, const int dstid) {
  if (CheckAlreadyExists(pq, *size, dstid)) return;
  const cuda_scalar* dst_vec = data + num_dims * dstid;
  cuda_scalar dist = GetDistanceByVec(src_vec, dst_vec, num_dims, dist_type);
  __syncthreads();
  if (*size < max_size) {
    PqPush(pq, size, dist, dstid, false);
  } else if (gt(pq[0].distance, dist)) {
    PqPop(pq, size);
    PqPush(pq, size, dist, dstid, false);
  }
  __syncthreads();
}


} // namespace cuhnsw

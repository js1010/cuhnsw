// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include "cuda_utils_kernels.cuh"

namespace cuhnsw {
__inline__ __device__
bool IsNeighbor(const int* graph, const int deg, const int dstid) {
  
  __syncthreads();
  // figure out the warp/ position inside the warp
  int warp =  threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  
  static __shared__ bool shared[WARP_SIZE];
  
  __syncthreads();
  bool is_neighbor = false;
  for (int i = threadIdx.x; i < deg; i += blockDim.x) {
    if (graph[i] == dstid) {
      is_neighbor = true;
      break;
    }
  }
  __syncthreads();
  
  #if __CUDACC_VER_MAJOR__ >= 9
  unsigned int active = __activemask();
  is_neighbor = __any_sync(active, is_neighbor);
  #else
  is_neighbor = __any(is_neighbor);
  #endif
  
  // write out the partial reduction to shared memory if appropiate
  if (lane == 0) {
    shared[warp] = is_neighbor;
  }
  
  __syncthreads();
  
  // if we we don't have multiple warps, we're done
  if (blockDim.x <= WARP_SIZE) {
    return shared[0];
  } 


  // otherwise reduce again in the first warp
  is_neighbor = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : false;
  if (warp == 0) {
    #if __CUDACC_VER_MAJOR__ >= 9
    active = __activemask();
    is_neighbor = __any_sync(active, is_neighbor);
    #else
    is_neighbor = __any(is_neighbor);
    #endif
    // broadcast back to shared memory
    if (threadIdx.x == 0) {
        shared[0] = is_neighbor;
    }
  }
  __syncthreads();
  return shared[0];
}

__inline__ __device__
void SearchHeuristic(
    Neighbor* ef_const_pq, int* size,
    const int srcid, const int* nodes,
    const cuda_scalar* data, const int dist_type, const int num_dims,
    const int ef_construction, const int max_m,
    const bool save_remains,
    int* cand_nodes, cuda_scalar* cand_distances, 
    int* graph, float* distances, int* deg, 
    const float heuristic_coef, const int new_comer = -1) {
  int size2 = *size;
  __syncthreads();

  // get sorted neighbors
  if (threadIdx.x == 0) {
    while (*size > 0) {
      cand_nodes[(*size) - 1] = ef_const_pq[0].nodeid;
      cand_distances[(*size) - 1] = ef_const_pq[0].distance;
      PqPop(ef_const_pq, size);
    }
  }
  __syncthreads();

  // set variables for search heuristic
  int head = 0;
  int tail = max_m - 1;
  if (tail > size2 - 1)
    tail = size2 - 1;
  const int max_head = tail + 1;
  
  // take some proportion of closest nodes by default
  // this mechanism does not exist in hnswlib
  // it refers to https://github.com/kakao/n2/blob/36888c3869ac478d896d0921ac64f21930d85659/src/heuristic.cc#L42
  const int nn_num = max_m * heuristic_coef;
  
  int* _graph = graph + srcid * max_m;
  float* _distances = distances + srcid * max_m;
  bool new_comer_inserted = false;
  // search heuristic
  for (int j = 0; j < size2; ++j) {
    if (head >= max_m) break;
    bool freepass = head < nn_num or 
      (new_comer >= 0 and not new_comer_inserted and cand_nodes[j] != new_comer);
    if (freepass) {
      if (threadIdx.x == 0) {
        _graph[head] = cand_nodes[j];
        _distances[head] = out_scalar(cand_distances[j]);
      }
      head++;
      __syncthreads();
      continue;
    }
    const cuda_scalar dist_to_src = cand_distances[j];
    bool skip = false;
    if (new_comer >= 0 and new_comer_inserted) {
      cuda_scalar dist = GetDistance(cand_nodes[j], new_comer,
          num_dims, dist_type, nodes, data);
      skip = gt(dist_to_src, dist);
    } else {
      for (int k = 0; k < head; ++k) {
        cuda_scalar dist = GetDistance(cand_nodes[j], _graph[k],
            num_dims, dist_type, nodes, data);
        if (gt(dist_to_src, dist)) {
          skip = true;
          __syncthreads();
          break;
        }
      }
    }
    if (cand_nodes[j] == new_comer and not skip) 
      new_comer_inserted = true;

    if (skip and tail >= head) {
      if (threadIdx.x == 0) {
        _graph[tail] = cand_nodes[j];
        _distances[tail] = out_scalar(cand_distances[j]);
      }
      tail--;
    } else if (not skip){
      if (threadIdx.x == 0) {
        _graph[head] = cand_nodes[j];
        _distances[head] = out_scalar(cand_distances[j]);
      }
      head++;
    }
  }
  __syncthreads();

  // copy to graph
  // take remaining nodes as new neighbors
  // it also refers to https://github.com/kakao/n2/blob/36888c3869ac478d896d0921ac64f21930d85659/src/heuristic.cc#L85
  // it does not exist in hnswlib as well
  if (threadIdx.x == 0) deg[srcid] = save_remains? max_head: head;
  __syncthreads();
}

__global__ void BuildLevelGraphKernel(
  const cuda_scalar* data, const int* nodes,
  const int num_dims, const int num_nodes, const int max_m, const int dist_type,
  const bool save_remains, const int ef_construction, int* graph, float* distances, int* deg,
  int* visited_table, int* visited_list, const int visited_table_size, const int visited_list_size, 
  int* mutex, int64_t* acc_visited_cnt,
  const bool reverse_cand, Neighbor* neighbors, int* global_cand_nodes, cuda_scalar* global_cand_distances,
  const float heuristic_coef, int* backup_neighbors, cuda_scalar* backup_distances, bool* went_through_heuristic
  ) {

  static __shared__ int size;
  static __shared__ int visited_cnt;
  
  // storage to store neighbors and distnces temporarily
  static __shared__ int backup_deg;
  int* _backup_neighbors = backup_neighbors + max_m * blockIdx.x;
  cuda_scalar* _backup_distances = backup_distances + max_m * blockIdx.x;

  Neighbor* ef_const_pq = neighbors + ef_construction * blockIdx.x;
  int* cand_nodes = global_cand_nodes + ef_construction * blockIdx.x;
  cuda_scalar* cand_distances = global_cand_distances + ef_construction * blockIdx.x;
  int* _visited_table = visited_table + visited_table_size * blockIdx.x;
  int* _visited_list = visited_list + visited_list_size * blockIdx.x;

  for (int i = blockIdx.x; i < num_nodes; i += gridDim.x) {
    if (threadIdx.x == 0) {
      size = 0;
      visited_cnt = 0;
    }
    __syncthreads();
    int srcid = i;
    // read access of srcid
    if (threadIdx.x == 0) {
      while (atomicCAS(&mutex[srcid], 0, 1)) {}
    }
    __syncthreads();

    // initialize entries as neighbors
    for (int j = max_m * i; j < max_m * i + deg[i]; ++j) {
      int dstid = graph[j];
      if (CheckVisited(_visited_table, _visited_list, visited_cnt, dstid, 
            visited_table_size, visited_list_size)) 
        continue;
      __syncthreads();

      PushNodeToPq(ef_const_pq, &size, ef_construction,
          data, num_dims, dist_type, srcid, dstid, nodes);
    }
    __syncthreads();

    // release lock
    if (threadIdx.x == 0) mutex[srcid] = 0;
    __syncthreads();

    // iterate until converge
    int idx = GetCand(ef_const_pq, size, reverse_cand);
    while (idx >= 0) {
      __syncthreads();
      if (threadIdx.x == 0) ef_const_pq[idx].checked = true;
      int entry = ef_const_pq[idx].nodeid;

      // read access of entry
      if (threadIdx.x == 0) {
        while (atomicCAS(&mutex[entry], 0, 1)) {}
      }
      __syncthreads();

      for (int j = max_m * entry; j < max_m * entry + deg[entry]; ++j) {
        int dstid = graph[j];

        if (CheckVisited(_visited_table, _visited_list, visited_cnt, dstid, 
              visited_table_size, visited_list_size)) 
          continue;
        __syncthreads();
        
        PushNodeToPq(ef_const_pq, &size, ef_construction,
            data, num_dims, dist_type, srcid, dstid, nodes);
      }
      __syncthreads();

      // release lock
      if (threadIdx.x == 0) mutex[entry] = 0;
      __syncthreads();
      idx = GetCand(ef_const_pq, size, reverse_cand);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      acc_visited_cnt[blockIdx.x] += visited_cnt;
    }
    for (int j = threadIdx.x; j < visited_cnt; j += blockDim.x) {
      _visited_table[_visited_list[j]] = -1;
    }
    __syncthreads();

    // write access of dstid
    if (threadIdx.x == 0) {
      while (atomicCAS(&mutex[srcid], 0, 1)) {}
    }
    __syncthreads();

    for (int j = 0; j < deg[srcid]; ++j) {
      int dstid = graph[srcid * max_m + j];
      PushNodeToPq(ef_const_pq, &size, ef_construction,
          data, num_dims, dist_type, srcid, dstid, nodes);
    }

    // run search heuristic for myself
    SearchHeuristic(ef_const_pq, &size, srcid, nodes,
        data, dist_type, num_dims,
        ef_construction, max_m, save_remains,
        cand_nodes, cand_distances, 
        graph, distances, deg, heuristic_coef);
    
    if (threadIdx.x == 0) went_through_heuristic[srcid] = true;

    __syncthreads();
    
    // backup neighbors to handle overwrite
    if (threadIdx.x == 0) backup_deg = deg[srcid];
    __syncthreads();
    for (int j = threadIdx.x; j < backup_deg; j += blockDim.x) {
      _backup_neighbors[j] = graph[srcid * max_m + j];
      _backup_distances[j] = conversion(distances[srcid * max_m + j]);
    }
    __syncthreads();
    // release lock
    if (threadIdx.x == 0) mutex[srcid] = 0;
    __syncthreads();

    // run search heuristic for neighbors
    for (int j = 0; j < backup_deg; ++j) {
      int dstid = _backup_neighbors[j];
      cuda_scalar dist = _backup_distances[j];
      __syncthreads();
      // write access of dstid
      if (threadIdx.x == 0) {
        while (atomicCAS(&mutex[dstid], 0, 1)) {}
      }
      __syncthreads();
      
      const int* _graph = graph + max_m * dstid;
      const int _deg = deg[dstid];
      bool is_neighbor = IsNeighbor(_graph, _deg, srcid);
      if (not is_neighbor){
        PushNodeToPq2(ef_const_pq, &size, ef_construction,
            dist, dstid, srcid, nodes);
        for (int k = 0; k < _deg; ++k) {
          int dstid2 = _graph[k];
          dist = conversion(distances[dstid * max_m + k]);
          PushNodeToPq2(ef_const_pq, &size, ef_construction,
              dist, dstid, dstid2, nodes);
        }

        __syncthreads();
        const int new_comer = not save_remains and went_through_heuristic[dstid]? srcid: -1;
        __syncthreads();
        SearchHeuristic(ef_const_pq, &size, dstid, nodes,
            data, dist_type, num_dims,
            ef_construction, max_m, save_remains,
            cand_nodes, cand_distances, 
            graph, distances, deg, heuristic_coef, new_comer);
        if (threadIdx.x == 0) went_through_heuristic[dstid] = true;
        __syncthreads();
      }
      // release lock
      if (threadIdx.x == 0) mutex[dstid] = 0;
      __syncthreads();
    }
    __syncthreads();
  }

  // cooperative_groups::grid_group g = cooperative_groups::this_grid();
  // g.sync();
}

} // namespace cuhnsw

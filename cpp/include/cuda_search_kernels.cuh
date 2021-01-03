// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include "cuda_utils_kernels.cuh"

namespace cuhnsw {

__global__ void GetEntryPointsKernel(
  const cuda_scalar* qdata, const int* qnodes, const cuda_scalar* target_data, const int* target_nodes,
  const int num_dims, const int num_qnodes, const int num_target_nodes, const int max_m, const int dist_type,
  const int* graph, const int* deg,
  bool* visited, int* visited_list, const int visited_list_size, int* entries, int64_t* acc_visited_cnt
  ) {

  static __shared__ int visited_cnt;
  bool* _visited = visited + num_target_nodes * blockIdx.x;
  int* _visited_list = visited_list + visited_list_size * blockIdx.x;

  for (int i = blockIdx.x; i < num_qnodes; i += gridDim.x) {
    if (threadIdx.x == 0) {
      visited_cnt = 0;
    }
    __syncthreads();
    cuda_scalar entry_dist = 0;
    int entryid = entries[i];
    const cuda_scalar* src_vec = qdata + num_dims * qnodes[i];
    {
      const cuda_scalar* dst_vec = target_data + num_dims * target_nodes[entryid];
      entry_dist = GetDistanceByVec(src_vec, dst_vec, num_dims, dist_type);
      // if (threadIdx.x == 0 and blockIdx.x == 0) { 
      //   printf("srcid: %d, dstid: %d, dist: %f\n", 
      //       qnodes[i], target_nodes[entryid], entry_dist);
      // }
    }
    __syncthreads();
    bool updated = true;
    while (updated) {
      // initialize entries as neighbors
      int beg = max_m * entryid;
      int end = beg + deg[entryid];
      updated = false;
      for (int j = beg; j < end; ++j) {
        int candid = graph[j];

        if (_visited[candid]) continue;
        __syncthreads();
        if (threadIdx.x == 0 and visited_cnt < visited_list_size) {
          _visited[candid] = true;
          _visited_list[visited_cnt++] = candid;
        }
        __syncthreads();
        const cuda_scalar* dst_vec = target_data + num_dims * target_nodes[candid];
        cuda_scalar dist = GetDistanceByVec(src_vec, dst_vec, num_dims, dist_type);
        if (dist < entry_dist) {
          entry_dist = dist;
          entryid = candid;
          updated = true;
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) entries[i] = entryid;
      __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      acc_visited_cnt[blockIdx.x] += visited_cnt;
    }
    for (int j = threadIdx.x; j < visited_cnt; j += blockDim.x) {
      _visited[_visited_list[j]] = false;
    }
    __syncthreads();
  }
}

__global__ void SearchGraphKernel(
  const cuda_scalar* qdata, const int num_qnodes, const cuda_scalar* data, const int num_nodes,
  const int num_dims, const int max_m, const int dist_type,
  const int ef_search, const int* entries, const int* graph, const int* deg, const int topk,
  int* nns, float* distances, int* found_cnt,
  int* visited_table, int* visited_list, 
  const int visited_table_size, const int visited_list_size, int64_t* acc_visited_cnt,
  const bool reverse_cand, Neighbor* neighbors, int* global_cand_nodes, cuda_scalar* global_cand_distances
  ) {

  static __shared__ int size;
  
  Neighbor* ef_search_pq = neighbors + ef_search * blockIdx.x;
  int* cand_nodes = global_cand_nodes + ef_search * blockIdx.x;
  cuda_scalar* cand_distances = global_cand_distances + ef_search * blockIdx.x;

  static __shared__ int visited_cnt;
  int* _visited_table = visited_table + visited_table_size * blockIdx.x;
  int* _visited_list = visited_list + visited_list_size * blockIdx.x;

  for (int i = blockIdx.x; i < num_qnodes; i += gridDim.x) {
    if (threadIdx.x == 0) {
      size = 0;
      visited_cnt = 0;
    }
    __syncthreads();

    // initialize entries
    const cuda_scalar* src_vec = qdata + i * num_dims;
    PushNodeToSearchPq(ef_search_pq, &size, ef_search, data, 
        num_dims, dist_type, src_vec, entries[i]);
    if (CheckVisited(_visited_table, _visited_list, visited_cnt, entries[i], 
          visited_table_size, visited_list_size)) 
      continue;
    __syncthreads();
    
    // iterate until converge
    int idx = GetCand(ef_search_pq, size, reverse_cand);
    while (idx >= 0) {
      __syncthreads();
      if (threadIdx.x == 0) ef_search_pq[idx].checked = true;
      int entry = ef_search_pq[idx].nodeid;
      __syncthreads();

      for (int j = max_m * entry; j < max_m * entry + deg[entry]; ++j) {
        int dstid = graph[j];

        if (CheckVisited(_visited_table, _visited_list, visited_cnt, dstid, 
              visited_table_size, visited_list_size)) 
          continue;
        __syncthreads();

        const cuda_scalar* dst_vec = data + num_dims * dstid;
        cuda_scalar dist = GetDistanceByVec(src_vec, dst_vec, num_dims, dist_type);

        PushNodeToSearchPq(ef_search_pq, &size, ef_search,
            data, num_dims, dist_type, src_vec, dstid);
      }
      __syncthreads();
      idx = GetCand(ef_search_pq, size, reverse_cand);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      acc_visited_cnt[blockIdx.x] += visited_cnt;
    }

    for (int j = threadIdx.x; j < visited_cnt; j += blockDim.x) {
      _visited_table[_visited_list[j]] = -1;
    }
    __syncthreads();
    // get sorted neighbors
    if (threadIdx.x == 0) {
      int size2 = size;
      while (size > 0) {
        cand_nodes[size - 1] = ef_search_pq[0].nodeid;
        cand_distances[size - 1] = ef_search_pq[0].distance;
        PqPop(ef_search_pq, &size);
      }
      found_cnt[i] = size2 < topk? size2: topk;
      for (int j = 0; j < found_cnt[i]; ++j) {
        nns[j + i * topk] = cand_nodes[j];
        distances[j + i * topk] = out_scalar(cand_distances[j]);
      }
    }
    __syncthreads();
  }
}



} // namespace cuhnsw

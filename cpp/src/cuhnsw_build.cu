// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include <iostream>
#include <algorithm>

#include "cuhnsw.hpp"
#include "cuda_search_kernels.cuh"
#include "cuda_build_kernels.cuh"

namespace cuhnsw {

void CuHNSW::GetDeviceInfo() {
  CHECK_CUDA(cudaGetDevice(&devId_));
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId_));
  mp_cnt_ = prop.multiProcessorCount;
  major_ = prop.major;
  minor_ = prop.minor;
  cores_ = -1;
}

void CuHNSW::GetEntryPoints(
    const std::vector<int>& nodes,
    std::vector<int>& entries,
    int level, bool search) {
  int size = nodes.size();
  
  // process input data for kernel
  LevelGraph& graph = level_graphs_[level];
  const std::vector<int>& upper_nodes = graph.GetNodes();
  int upper_size = upper_nodes.size();
  std::vector<int> deg(upper_size);
  std::vector<int> neighbors(upper_size * max_m_);
  for (int i = 0; i < upper_size; ++i) {
    const std::vector<std::pair<float, int>>& _neighbors = graph.GetNeighbors(upper_nodes[i]);
    deg[i] = _neighbors.size();
    int offset = max_m_ * i;
    for (int j = 0; j < deg[i]; ++j) {
      neighbors[offset + j] = graph.GetNodeId(_neighbors[j].second);
    }
  }
  for (int i = 0; i < size; ++i) {
    int entryid = graph.GetNodeId(entries[i]); 
    entries[i] = entryid;
  }
  
  // copy to gpu mem
  thrust::device_vector<int> dev_nodes(size), dev_entries(size);
  thrust::device_vector<int> dev_upper_nodes(upper_size), dev_deg(upper_size);
  thrust::device_vector<int> dev_neighbors(upper_size * max_m_);
  thrust::copy(nodes.begin(), nodes.end(), dev_nodes.begin());
  thrust::copy(entries.begin(), entries.end(), dev_entries.begin());
  thrust::copy(upper_nodes.begin(), upper_nodes.end(), dev_upper_nodes.begin());
  thrust::copy(deg.begin(), deg.end(), dev_deg.begin());
  thrust::copy(neighbors.begin(), neighbors.end(), dev_neighbors.begin());
  
  thrust::device_vector<bool> dev_visited(upper_size * block_cnt_, false);
  thrust::device_vector<int> dev_visited_list(visited_list_size_ * block_cnt_);
  thrust::device_vector<int64_t> dev_acc_visited_cnt(block_cnt_, 0);
  thrust::device_vector<cuda_scalar>& qdata = search? device_qdata_: device_data_;

  // run kernel
  GetEntryPointsKernel<<<block_cnt_, block_dim_>>>(
    thrust::raw_pointer_cast(qdata.data()),
    thrust::raw_pointer_cast(dev_nodes.data()),
    thrust::raw_pointer_cast(device_data_.data()),
    thrust::raw_pointer_cast(dev_upper_nodes.data()),
    num_dims_, size, upper_size, max_m_, dist_type_,
    thrust::raw_pointer_cast(dev_neighbors.data()),
    thrust::raw_pointer_cast(dev_deg.data()),
    thrust::raw_pointer_cast(dev_visited.data()),
    thrust::raw_pointer_cast(dev_visited_list.data()),
    visited_list_size_,
    thrust::raw_pointer_cast(dev_entries.data()),
    thrust::raw_pointer_cast(dev_acc_visited_cnt.data())
    );
  CHECK_CUDA(cudaDeviceSynchronize());
  // el_[GPU] += sw_[GPU].CheckPoint();
  thrust::copy(dev_entries.begin(), dev_entries.end(), entries.begin());
  std::vector<int64_t> acc_visited_cnt(block_cnt_);
  thrust::copy(dev_acc_visited_cnt.begin(), dev_acc_visited_cnt.end(), acc_visited_cnt.begin());
  CHECK_CUDA(cudaDeviceSynchronize());
  int64_t full_visited_cnt = std::accumulate(acc_visited_cnt.begin(), acc_visited_cnt.end(), 0);
  DEBUG("full visited cnt: {}", full_visited_cnt);
  
  // set output
  for (int i = 0; i < size; ++i) {
    entries[i] = upper_nodes[entries[i]];
  }

}

void CuHNSW::BuildGraph() {
  visited_ = new bool[batch_size_ * num_data_];
  for (int level = max_level_; level >= 0; --level) {
    DEBUG("build graph of level: {}", level);
    BuildLevelGraph(level);
  }
}

void CuHNSW::BuildLevelGraph(int level) {
  std::set<int> upper_nodes;
  std::vector<int> new_nodes;
  LevelGraph& graph = level_graphs_[level];
  const std::vector<int>& nodes = graph.GetNodes();
  int size = nodes.size();
  int max_m = level > 0? max_m_: max_m0_;
  thrust::host_vector<int> graph_vec(size * max_m, 0);
  thrust::host_vector<int> deg(size, 0);
  if (level < max_level_) {
    LevelGraph& upper_graph = level_graphs_[level + 1];
    for (auto& node: upper_graph.GetNodes()) {
      upper_nodes.insert(node);
      int srcid = graph.GetNodeId(node);
      int idx = 0;
      for (auto& nb: upper_graph.GetNeighbors(node)) {
        int dstid = graph.GetNodeId(nb.second);
        graph_vec[max_m * srcid + (idx++)] = dstid;
      }
      deg[srcid] = idx;
    }
  }

  for (auto& node: graph.GetNodes()) {
    if (upper_nodes.count(node)) continue;
    new_nodes.push_back(node);
  }
  
  // initialize entries
  std::vector<int> entries(new_nodes.size(), enter_point_);

  for (int l = max_level_; l > level; --l)
    GetEntryPoints(new_nodes, entries, l, false);
  for (int i = 0; i < new_nodes.size(); ++i) {
    int srcid = graph.GetNodeId(new_nodes[i]);
    int dstid = graph.GetNodeId(entries[i]);
    graph_vec[max_m * srcid] = dstid;
    deg[srcid] = 1;
  }

  thrust::device_vector<int> device_graph(max_m * size);
  thrust::device_vector<float> device_distances(max_m * size);
  thrust::device_vector<int> device_deg(size);
  thrust::device_vector<int> device_nodes(size);
  thrust::device_vector<int> device_visited_table(visited_table_size_ * block_cnt_, -1);
  thrust::device_vector<int> device_visited_list(visited_list_size_ * block_cnt_);
  thrust::device_vector<int> device_mutex(size, 0);
  thrust::device_vector<int64_t> device_acc_visited_cnt(block_cnt_, 0);
  thrust::device_vector<Neighbor> device_neighbors(ef_construction_ * block_cnt_);
  thrust::device_vector<int> device_cand_nodes(ef_construction_ * block_cnt_);
  thrust::device_vector<cuda_scalar> device_cand_distances(ef_construction_ * block_cnt_);
  thrust::device_vector<int> device_backup_neighbors(max_m * block_cnt_);
  thrust::device_vector<cuda_scalar> device_backup_distances(max_m * block_cnt_);
  thrust::device_vector<bool> device_went_through_heuristic(size, false);

  thrust::copy(graph_vec.begin(), graph_vec.end(), device_graph.begin());
  thrust::copy(deg.begin(), deg.end(), device_deg.begin());
  thrust::copy(nodes.begin(), nodes.end(), device_nodes.begin());

  BuildLevelGraphKernel<<<block_cnt_, block_dim_>>>(
    thrust::raw_pointer_cast(device_data_.data()),
    thrust::raw_pointer_cast(device_nodes.data()),
    num_dims_, size, max_m, dist_type_, save_remains_,
    ef_construction_,
    thrust::raw_pointer_cast(device_graph.data()),
    thrust::raw_pointer_cast(device_distances.data()),
    thrust::raw_pointer_cast(device_deg.data()),
    thrust::raw_pointer_cast(device_visited_table.data()),
    thrust::raw_pointer_cast(device_visited_list.data()),
    visited_table_size_, visited_list_size_,
    thrust::raw_pointer_cast(device_mutex.data()),
    thrust::raw_pointer_cast(device_acc_visited_cnt.data()),
    reverse_cand_,
    thrust::raw_pointer_cast(device_neighbors.data()),
    thrust::raw_pointer_cast(device_cand_nodes.data()),
    thrust::raw_pointer_cast(device_cand_distances.data()),
    heuristic_coef_,
    thrust::raw_pointer_cast(device_backup_neighbors.data()),
    thrust::raw_pointer_cast(device_backup_distances.data()),
    thrust::raw_pointer_cast(device_went_through_heuristic.data())
    );
  CHECK_CUDA(cudaDeviceSynchronize());
  thrust::copy(device_deg.begin(), device_deg.end(), deg.begin());
  thrust::copy(device_graph.begin(), device_graph.end(), graph_vec.begin());
  std::vector<float> distances(max_m * size);
  thrust::copy(device_distances.begin(), device_distances.end(), distances.begin());

  std::vector<int64_t> acc_visited_cnt(block_cnt_);
  thrust::copy(device_acc_visited_cnt.begin(), device_acc_visited_cnt.end(), acc_visited_cnt.begin());
  CHECK_CUDA(cudaDeviceSynchronize());
  int64_t full_visited_cnt = std::accumulate(acc_visited_cnt.begin(), acc_visited_cnt.end(), 0LL);
  DEBUG("full number of visited nodes: {}", full_visited_cnt);

  for (auto& node: graph.GetNodes()) {
    graph.ClearEdges(node);
  }
  for (int i = 0; i < size; ++i) {
    int src = nodes[i];
    for (int j = 0; j < deg[i]; ++j) {
      int dst = nodes[graph_vec[i * max_m + j]];
      float dist = distances[i * max_m + j];
      graph.AddEdge(src, dst, dist);
    }
  }
}

void CuHNSW::SearchGraph(const float* qdata, const int num_queries, const int topk, const int ef_search, 
    int* nns, float* distances, int* found_cnt) {
  device_qdata_.resize(num_queries * num_dims_);
  #ifdef HALF_PRECISION
    std::vector<cuda_scalar> hdata(num_queries * num_dims_);
    for (int i = 0; i < num_queries * num_dims_; ++i)
      hdata[i] = conversion(qdata[i]);
    thrust::copy(hdata.begin(), hdata.end(), device_qdata_.begin());
  #else
    thrust::copy(qdata, qdata + num_queries * num_dims_, device_qdata_.begin());
  #endif
  std::vector<int> qnodes(num_queries);
  std::iota(qnodes.begin(), qnodes.end(), 0);
  std::vector<int> entries(num_queries, enter_point_);
  for (int l = max_level_; l > 0; --l)
    GetEntryPoints(qnodes, entries, l, true);
  std::vector<int> graph_vec(max_m0_ * num_data_);
  std::vector<int> deg(num_data_);
  LevelGraph graph = level_graphs_[0];
  for (int i = 0; i < num_data_; ++i) {
    const std::vector<std::pair<float, int>>& neighbors = graph.GetNeighbors(i);
    int nbsize = neighbors.size();
    int offset = i * max_m0_;
    for (int j = 0; j < nbsize; ++j)
      graph_vec[offset + j] = neighbors[j].second;
    deg[i] = nbsize;
  }
  
  thrust::device_vector<int> device_graph(max_m0_ * num_data_);
  thrust::device_vector<int> device_deg(num_data_);
  thrust::device_vector<int> device_entries(num_queries);
  thrust::device_vector<int> device_nns(num_queries * topk);
  thrust::device_vector<float> device_distances(num_queries * topk);
  thrust::device_vector<int> device_found_cnt(num_queries);
  thrust::device_vector<int> device_visited_table(visited_table_size_ * block_cnt_, -1);
  thrust::device_vector<int> device_visited_list(visited_list_size_ * block_cnt_);
  thrust::device_vector<int64_t> device_acc_visited_cnt(block_cnt_, 0);
  thrust::device_vector<Neighbor> device_neighbors(ef_search * block_cnt_);
  thrust::device_vector<int> device_cand_nodes(ef_search * block_cnt_);
  thrust::device_vector<cuda_scalar> device_cand_distances(ef_search * block_cnt_);

  thrust::copy(graph_vec.begin(), graph_vec.end(), device_graph.begin());
  thrust::copy(deg.begin(), deg.end(), device_deg.begin());
  thrust::copy(entries.begin(), entries.end(), device_entries.begin());
  SearchGraphKernel<<<block_cnt_, block_dim_>>>(
    thrust::raw_pointer_cast(device_qdata_.data()),
    num_queries,
    thrust::raw_pointer_cast(device_data_.data()),
    num_data_, num_dims_, max_m0_, dist_type_, ef_search, 
    thrust::raw_pointer_cast(device_entries.data()),
    thrust::raw_pointer_cast(device_graph.data()),
    thrust::raw_pointer_cast(device_deg.data()), 
    topk,
    thrust::raw_pointer_cast(device_nns.data()), 
    thrust::raw_pointer_cast(device_distances.data()), 
    thrust::raw_pointer_cast(device_found_cnt.data()), 
    thrust::raw_pointer_cast(device_visited_table.data()),
    thrust::raw_pointer_cast(device_visited_list.data()),
    visited_table_size_, visited_list_size_,
    thrust::raw_pointer_cast(device_acc_visited_cnt.data()),
    reverse_cand_,
    thrust::raw_pointer_cast(device_neighbors.data()),
    thrust::raw_pointer_cast(device_cand_nodes.data()),
    thrust::raw_pointer_cast(device_cand_distances.data())
    );
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<int64_t> acc_visited_cnt(block_cnt_);
  thrust::copy(device_acc_visited_cnt.begin(), device_acc_visited_cnt.end(), acc_visited_cnt.begin());
  thrust::copy(device_nns.begin(), device_nns.end(), nns);
  thrust::copy(device_distances.begin(), device_distances.end(), distances);
  thrust::copy(device_found_cnt.begin(), device_found_cnt.end(), found_cnt);
  CHECK_CUDA(cudaDeviceSynchronize());
  int64_t full_visited_cnt = std::accumulate(acc_visited_cnt.begin(), acc_visited_cnt.end(), 0LL);
  DEBUG("full number of visited nodes: {}", full_visited_cnt);
  if (labelled_)
    for (int i = 0; i < num_queries * topk; ++i)
      nns[i] = labels_[nns[i]];
  
  device_qdata_.clear();
  device_qdata_.shrink_to_fit();
}

} // namespace cuhnsw

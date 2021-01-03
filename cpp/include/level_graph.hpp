// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include <set>
#include <unordered_set>
#include <random>
#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <queue>
#include <functional>
#include <vector>
#include <unordered_map>

#include "log.hpp"

namespace cuhnsw {

class LevelGraph {
 public:
  LevelGraph() {
    logger_ = CuHNSWLogger().get_logger();
  }

  ~LevelGraph() {}

  void SetNodes(std::vector<int>& nodes, int num_data, int ef_construction) {
    nodes_ = nodes;
    num_nodes_ = nodes_.size();
    neighbors_.clear();
    neighbors_.resize(num_nodes_);
    nodes_idmap_.resize(num_data);
    std::fill(nodes_idmap_.begin(), nodes_idmap_.end(), -1);
    for (int i = 0; i < num_nodes_; ++i)
      nodes_idmap_[nodes[i]] = i;
  }

  const std::vector<std::pair<float, int>>& GetNeighbors(int node) const  {
    int nodeid = GetNodeId(node);
    return neighbors_[nodeid];
  }

  const std::vector<int>& GetNodes() const {
    return nodes_;
  }

  void ClearEdges(int node) {
    neighbors_[GetNodeId(node)].clear();
  }

  void AddEdge(int src, int dst, float dist) {
    if (src == dst) return;
    int srcid = GetNodeId(src);
    neighbors_[srcid].emplace_back(dist, dst);
  }

  inline int GetNodeId(int node) const {
    int nodeid = nodes_idmap_.at(node);
    if (not(nodeid >= 0 and nodeid < num_nodes_)) {
      throw std::runtime_error(
          fmt::format("[{}:{}] invalid nodeid: {}, node: {}, num_nodes: {}",
            __FILE__, __LINE__, nodeid, node, num_nodes_));
    }
    return nodeid;
  }

  void ShowGraph() {
    for (int i = 0; i < num_nodes_; ++i) {
      std::cout << std::string(50, '=') << std::endl;
      printf("nodeid %d: %d\n", i, nodes_[i]);
      for (auto& nb: GetNeighbors(nodes_[i])) {
        printf("neighbor id: %d, dist: %f\n",
            nb.second, nb.first);
      }
      std::cout << std::string(50, '=') << std::endl;
    }
  }

 private:
  std::shared_ptr<spdlog::logger> logger_;
  std::vector<int> nodes_;
  std::vector<std::vector<std::pair<float, int>>> neighbors_;
  int num_nodes_ = 0;
  std::vector<int> nodes_idmap_;
};  // class LevelGraph

} // namespace cuhnsw

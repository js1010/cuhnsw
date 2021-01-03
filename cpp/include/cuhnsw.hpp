// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <omp.h>
#include <set>
#include <random>
#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <queue>
#include <deque>
#include <functional>
#include <vector>
#include <cmath>
#include <chrono> // NOLINT

#include "json11.hpp"
#include "log.hpp"
#include "level_graph.hpp"
// #include "stop_watch.hpp"
#include "types.hpp"

namespace cuhnsw {

// for the compatibility with hnswlib
// following two functions refer to
// https://github.com/nmslib/hnswlib/blob/
// 2571bdb6ef3f91d6f4c2e59178fde49055d2f980/hnswlib/hnswlib.h
template<typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write(reinterpret_cast<const char*>(&podRef), sizeof(T));
}
template<typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read(reinterpret_cast<char*>(&podRef), sizeof(T));
}

class CuHNSW {
 public:
  // enum ProfileColumns {
  //   GPU,
  //   PROFILE_SIZE,
  // };

  // std::vector<std::string> PROFILE_KEYS = {
  //   "gpu",
  // };

  CuHNSW();
  ~CuHNSW();

  bool Init(std::string opt_path);
  void SetData(const float* data, int num_data, int num_dims);
  void SetRandomLevels(const int* levels);
  void BuildGraph();
  void SaveIndex(std::string fpath);
  void LoadIndex(std::string fpath);
  void SearchGraph(const float* qdata, const int num_queries, const int topk, const int ef_search,
    int* nns, float* distances, int* found_cnt);

 private:
  void GetDeviceInfo();
  void GetEntryPoints(
      const std::vector<int>& nodes,
      std::vector<int>& entries,
      int level, bool search);
  void SearchAtLayer(
      const std::vector<int>& queries,
      std::vector<std::deque<std::pair<float, int>>>& entries,
      int level, int max_m);
  void SearchHeuristicAtLayer(
      const std::vector<int>& queries,
      int level, int max_m, bool postprocess);
  void BuildLevelGraph(int level);
  std::vector<LevelGraph> level_graphs_;
  std::vector<int> levels_;

  json11::Json opt_;
  std::shared_ptr<spdlog::logger> logger_;

  int num_data_, num_dims_, batch_size_;
  thrust::device_vector<cuda_scalar> device_data_, device_qdata_;
  const float* data_;
  std::vector<int> labels_;
  bool labelled_ = false;
  bool reverse_cand_ = false;

  int major_, minor_, cores_, devId_, mp_cnt_;
  int block_cnt_, block_dim_;
  int visited_table_size_, visited_list_size_;
  int max_level_, max_m_, max_m0_;
  int enter_point_, ef_construction_;
  float level_mult_;
  int dist_type_;
  bool save_remains_;
  double heuristic_coef_;
  // std::vector<StopWatch> sw_;
  // std::vector<double> el_;

  bool* visited_;
};  // class CuHNSW

} // namespace cuhnsw

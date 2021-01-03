// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include <iostream>
#include <algorithm>

#include "cuhnsw.hpp"

namespace cuhnsw {

CuHNSW::CuHNSW() {
  logger_ = CuHNSWLogger().get_logger();

  GetDeviceInfo();
  // reference: https://stackoverflow.com/a/32531982
  switch (major_){
    case 2: // Fermi
      if (minor_ == 1) 
        cores_ = mp_cnt_ * 48;
      else 
        cores_ = mp_cnt_ * 32;
      break;
    case 3: // Kepler
      cores_ = mp_cnt_ * 192;
      break;
    case 5: // Maxwell
      cores_ = mp_cnt_ * 128;
      break;
    case 6: // Pascal
      if (minor_ == 1 or minor_ == 2) 
        cores_ = mp_cnt_ * 128;
      else if (minor_ == 0) 
        cores_ = mp_cnt_ * 64;
      else 
        DEBUG0("Unknown device type");
      break;
    case 7: // Volta and Turing
      if (minor_ == 0 or minor_ == 5) 
        cores_ = mp_cnt_ * 64;
      else 
        DEBUG0("Unknown device type");
      break;
    case 8: // Ampere
      if (minor_ == 0) 
        cores_ = mp_cnt_ * 64;
      else if (minor_ == 6) 
        cores_ = mp_cnt_ * 128;
      else 
        DEBUG0("Unknown device type");
      break;
    default:
      DEBUG0("Unknown device type"); 
      break;
  }
  if (cores_ == -1) cores_ = mp_cnt_ * 128;
  INFO("cuda device info, major: {}, minor: {}, multi processors: {}, cores: {}",
       major_, minor_, mp_cnt_, cores_);
  // sw_.resize(PROFILE_SIZE);
  // el_.resize(PROFILE_SIZE);
}


CuHNSW::~CuHNSW() {}

bool CuHNSW::Init(std::string opt_path) {
  std::ifstream in(opt_path.c_str());
  if (not in.is_open()) return false;

  std::string str((std::istreambuf_iterator<char>(in)),
      std::istreambuf_iterator<char>());
  std::string err_cmt;
  auto _opt = json11::Json::parse(str, err_cmt);
  if (not err_cmt.empty()) return false;
  opt_ = _opt;
  max_m_ = opt_["max_m"].int_value();
  max_m0_ = opt_["max_m0"].int_value();
  save_remains_ = opt_["save_remains"].bool_value();
  ef_construction_ = opt_["ef_construction"].int_value();
  level_mult_ = opt_["level_mult"].number_value();
  batch_size_ = opt_["batch_size"].int_value();
  block_dim_ = opt_["block_dim"].int_value();
  visited_table_size_ = opt_["visited_table_size"].int_value();
  visited_list_size_ = opt_["visited_list_size"].int_value();
  if (not visited_table_size_)
    visited_table_size_ = visited_list_size_ * 2;
  heuristic_coef_ = opt_["heuristic_coef"].number_value();
  std::string dist_type = opt_["dist_type"].string_value();
  reverse_cand_ = opt_["reverse_cand"].bool_value();
  if (dist_type == "dot") {
    dist_type_ = DOT;
  } else if (dist_type == "l2") {
    dist_type_ = L2;
  } else {
    char buf[4096];
    snprintf(buf, sizeof(buf), "invalid dist type %s",
        dist_type.c_str());
    std::string msg(buf);
    throw std::runtime_error(msg);
  }
  CuHNSWLogger().set_log_level(opt_["c_log_level"].int_value());
  DEBUG("max_m: {}, max_m0: {}, save_remains: {}, ef_construction: {}, level_mult: {}, dist_type: {}",
      max_m_, max_m0_, save_remains_, ef_construction_, level_mult_, dist_type);
  return true;
}

void CuHNSW::SetData(const float* data, int num_data, int num_dims) {
  num_data_ = num_data;
  num_dims_ = num_dims;
  block_cnt_ = opt_["hyper_threads"].number_value() * (cores_ / block_dim_);
  DEBUG("copy data ({} x {}), block_cnt: {}, block_dim: {}",
      num_data, num_dims, block_cnt_, block_dim_);
  device_data_.resize(num_data * num_dims);
  #ifdef HALF_PRECISION
    // DEBUG0("fp16")
    std::vector<cuda_scalar> hdata(num_data * num_dims);
    for (int i = 0; i < num_data * num_dims; ++i) {
      hdata[i] = conversion(data[i]);
      // DEBUG("hdata i: {}, scalar: {}", i, out_scalar(hdata[i]));
    }
    thrust::copy(hdata.begin(), hdata.end(), device_data_.begin());
  #else
    // DEBUG0("fp32")
    thrust::copy(data, data + num_data * num_dims, device_data_.begin());
  #endif
  data_ = data;
}

void CuHNSW::SetRandomLevels(const int* levels) {
  levels_.resize(num_data_);
  DEBUG("set levels of data (length: {})", num_data_)
  max_level_ = 0;
  std::vector<std::vector<int>> level_nodes(1);
  for (int i = 0; i < num_data_; ++i) {
    levels_[i] = levels[i];
    if (levels[i] > max_level_) {
      max_level_ = levels[i];
      level_nodes.resize(max_level_ + 1);
      enter_point_ = i;
    }
    for (int l = 0; l <= levels[i]; ++l)
      level_nodes[l].push_back(i);
  }
  DEBUG("max level: {}", max_level_)
  for (int i = 0; i <= max_level_; ++i)
    DEBUG("number of data in level {}: {}",
        i, level_nodes[i].size());
  level_graphs_.clear();
  for (int i = 0; i <= max_level_; ++i) {
    LevelGraph graph = LevelGraph();
    graph.SetNodes(level_nodes[i],
        num_data_, ef_construction_);
    level_graphs_.push_back(graph);
  }
}

// save graph compatible with hnswlib (https://github.com/nmslib/hnswlib)
void CuHNSW::SaveIndex(std::string fpath) {
  std::ofstream output(fpath);
  DEBUG("save index to {}", fpath);
  
  // write meta values
  DEBUG0("write meta values"); 
  size_t data_size = num_dims_ * sizeof(scalar);
  size_t max_elements = num_data_;
  size_t cur_element_count = num_data_;
  size_t M = max_m_;
  size_t maxM = max_m_;
  size_t maxM0 = max_m0_;
  int maxlevel = max_level_;
  size_t size_links_level0 = maxM0 * sizeof(tableint) + sizeof(sizeint);
  size_t size_links_per_element = maxM * sizeof(tableint) + sizeof(sizeint);
  size_t size_data_per_element = size_links_level0 + data_size + sizeof(labeltype);
  size_t ef_construction = ef_construction_; 
  double mult = level_mult_; 
  size_t offsetData = size_links_level0;
  size_t label_offset = size_links_level0 + data_size;
  size_t offsetLevel0 = 0;
  tableint enterpoint_node = enter_point_;
  
  writeBinaryPOD(output, offsetLevel0);
  writeBinaryPOD(output, max_elements);
  writeBinaryPOD(output, cur_element_count);
  writeBinaryPOD(output, size_data_per_element);
  writeBinaryPOD(output, label_offset);
  writeBinaryPOD(output, offsetData);
  writeBinaryPOD(output, maxlevel);
  writeBinaryPOD(output, enterpoint_node);
  writeBinaryPOD(output, maxM);
  writeBinaryPOD(output, maxM0);
  writeBinaryPOD(output, M);
  writeBinaryPOD(output, mult);
  writeBinaryPOD(output, ef_construction);

  // write level0 links and data
  DEBUG0("write level0 links and data"); 
  char* data_level0_memory = (char*) malloc(cur_element_count * size_data_per_element);
  LevelGraph& graph = level_graphs_[0];
  std::vector<tableint> links;
  links.reserve(max_m0_);
  size_t offset = 0;
  for (int i = 0; i < cur_element_count; ++i) {
    links.clear();
    for (const auto& pr: graph.GetNeighbors(i))
      links.push_back(static_cast<tableint>(pr.second));
    
    sizeint size = links.size();
    memcpy(data_level0_memory + offset, &size, sizeof(sizeint));
    offset += sizeof(sizeint);
    if (size > 0)
      memcpy(data_level0_memory + offset, &links[0], sizeof(tableint) * size);
    offset += maxM0 * sizeof(tableint); 
    memcpy(data_level0_memory + offset, &data_[i * num_dims_], data_size);
    offset += data_size;
    labeltype label = i;
    memcpy(data_level0_memory + offset, &label, sizeof(labeltype));
    offset += sizeof(labeltype);
  }
  output.write(data_level0_memory, cur_element_count * size_data_per_element);
  
  // write upper layer links
  DEBUG0("write upper layer links");
  for (int i = 0; i < num_data_; ++i) {
    unsigned int size = size_links_per_element * levels_[i];
    writeBinaryPOD(output, size);
    char* mem = (char*) malloc(size);
    offset = 0;
    if (size) {
      for (int j = 1; j <= levels_[i]; ++j) {
        links.clear();
        LevelGraph& upper_graph = level_graphs_[j];
        for (const auto& pr: upper_graph.GetNeighbors(i))
          links.push_back(static_cast<tableint>(pr.second));
        sizeint link_size = links.size();
        memcpy(mem + offset, &link_size, sizeof(sizeint));
        offset += sizeof(sizeint);
        if (link_size > 0)
          memcpy(mem + offset, &links[0], sizeof(tableint) * link_size);
        offset += sizeof(tableint) * maxM;
      }
      output.write(mem, size);
    }
  }

  output.close();
}

// load graph compatible with hnswlib (https://github.com/nmslib/hnswlib)
void CuHNSW::LoadIndex(std::string fpath) {
  std::ifstream input(fpath, std::ios::binary);
  DEBUG("load index from {}", fpath);
  
  // reqd meta values
  DEBUG0("read meta values"); 
  size_t offsetLevel0, max_elements, cur_element_count;
  size_t size_data_per_element, label_offset, offsetData;
  int maxlevel; 
  tableint enterpoint_node = enter_point_;
  size_t maxM, maxM0, M;
  double mult;
  size_t ef_construction;

  readBinaryPOD(input, offsetLevel0);
  readBinaryPOD(input, max_elements);
  readBinaryPOD(input, cur_element_count);
  readBinaryPOD(input, size_data_per_element);
  readBinaryPOD(input, label_offset);
  readBinaryPOD(input, offsetData);
  readBinaryPOD(input, maxlevel);
  readBinaryPOD(input, enterpoint_node);
  readBinaryPOD(input, maxM);
  readBinaryPOD(input, maxM0);
  readBinaryPOD(input, M);
  readBinaryPOD(input, mult);
  readBinaryPOD(input, ef_construction);
  size_t size_per_link = maxM * sizeof(tableint) + sizeof(sizeint);
  num_data_ = cur_element_count;
  max_m_ = maxM;
  max_m0_ = maxM0;
  enter_point_ = enterpoint_node;
  ef_construction_ = ef_construction;
  max_level_ = maxlevel;
  level_mult_ = mult;
  num_dims_ = (label_offset - offsetData) / sizeof(scalar);
  DEBUG("meta values loaded, num_data: {}, num_dims: {}, max_m: {}, max_m0: {}, enter_point: {}, max_level: {}",
      num_data_, num_dims_, max_m_, max_m0_, enter_point_, max_level_);

  char* data_level0_memory = (char*) malloc(max_elements * size_data_per_element);
  input.read(data_level0_memory, cur_element_count * size_data_per_element);
  
  // reset level graphs
  level_graphs_.clear();
  level_graphs_.shrink_to_fit();
  level_graphs_.resize(max_level_ + 1);
  
  // load data and level0 links
  DEBUG0("load level0 links and data");
  DEBUG("level0 count: {}", cur_element_count);
  std::vector<float> data(num_data_ * num_dims_);
  size_t offset = 0;
  std::vector<tableint> links(max_m0_);
  std::vector<scalar> vec_data(num_dims_);
  LevelGraph& graph0 = level_graphs_[0];
  std::vector<std::vector<int>> nodes(max_level_ + 1);
  nodes[0].resize(cur_element_count);
  std::iota(nodes[0].begin(), nodes[0].end(), 0);
  graph0.SetNodes(nodes[0], num_data_, ef_construction_);
  labels_.clear(); labelled_ = true;
  for (int i = 0; i < cur_element_count; ++i) {
    sizeint deg;
    memcpy(&deg, data_level0_memory + offset, sizeof(sizeint));
    offset += sizeof(sizeint);
    memcpy(&links[0], data_level0_memory + offset, sizeof(tableint) * max_m0_);
    for (int j = 0; j < deg; ++j)
      graph0.AddEdge(i, links[j], 0);
    offset += sizeof(tableint) * max_m0_;
    memcpy(&vec_data[0], data_level0_memory + offset, sizeof(scalar) * num_dims_);
    for (int j = 0; j < num_dims_; ++j)
      data[num_dims_ * i + j] = vec_data[j];
    offset += sizeof(scalar) * num_dims_;
    labeltype label;
    memcpy(&label, data_level0_memory + offset, sizeof(labeltype));
    labels_.push_back(static_cast<int>(label));
    offset += sizeof(labeltype);
  }
  SetData(&data[0], num_data_, num_dims_);
  
  // load upper layer links
  DEBUG0("load upper layer links");
  std::vector<std::vector<std::pair<int, int>>> links_data(max_level_ + 1);
  links.resize(max_m_);
  levels_.resize(cur_element_count);
  for (int i = 0; i < cur_element_count; ++i) {
    unsigned int linksize;
    readBinaryPOD(input, linksize);
    if (not linksize) continue;
    char* buffer = (char*) malloc(linksize);
    input.read(buffer, linksize);
    size_t levels = linksize / size_per_link;
    size_t offset = 0;
    levels_[i] = levels + 1;
    for (int j = 1; j <= levels; ++j) {
      nodes[j].push_back(i);
      sizeint deg;
      memcpy(&deg, buffer + offset, sizeof(sizeint));
      offset += sizeof(sizeint);
      memcpy(&links[0], buffer + offset, sizeof(tableint) * deg);
      offset += sizeof(tableint) * max_m_;
      for (int k = 0; k < deg; ++k)
        links_data[j].emplace_back(i, links[k]); 
    }
  }

  for (int i = 1; i <= max_level_; ++i) {
    LevelGraph& graph = level_graphs_[i];
    DEBUG("level {} count: {}", i, nodes[i].size());
    graph.SetNodes(nodes[i], num_data_, ef_construction_);
    for (const auto& pr: links_data[i]) {
      graph.AddEdge(pr.first, pr.second, 0);
    }
  }

  input.close();
}

} // namespace cuhnsw

// Copyright (c) 2020 Jisang Yoon
//  All rights reserved.
//
//  This source code is licensed under the Apache 2.0 license found in the
//  LICENSE file in the root directory of this source tree.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include "cuhnsw.hpp"

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> float_array;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> int_array;

class CuHNSWBind {
 public:
  CuHNSWBind() {}

  bool Init(std::string opt_path) {
    return obj_.Init(opt_path);
  }

  void SetData(py::object& input) {
    float_array array(input);
    auto buffer = array.request();
    if (buffer.ndim != 2) throw std::runtime_error("data must be 2d array");
    int num_data = buffer.shape[0];
    int num_dims = buffer.shape[1];
    obj_.SetData(array.data(0), num_data, num_dims);
  }

  void BuildGraph() {
    obj_.BuildGraph();
  }

  void SetRandomLevels(py::object& input) {
    int_array array(input);
    auto buffer = array.request();
    if (buffer.ndim != 1) throw std::runtime_error("levels must be 1d array");
    obj_.SetRandomLevels(array.data(0));
  }

  void SaveIndex(std::string fpath) {
    obj_.SaveIndex(fpath);
  }

  void LoadIndex(std::string fpath) {
    obj_.LoadIndex(fpath);
  }

  void SearchGraph(py::object& qdata, int topk, int ef_search,
      py::object& nns, py::object& distances, py::object& found_cnt) {
    float_array _qdata(qdata);
    int_array _nns(nns);
    float_array _distances(distances);
    int_array _found_cnt(found_cnt);
    auto buffer = _qdata.request();

    if (buffer.ndim != 1 and buffer.ndim != 2)
      throw std::runtime_error("data array must be 1d / 2d shape");

    int num_queries = buffer.ndim == 1? 1: buffer.shape[0];
    obj_.SearchGraph(_qdata.data(0), num_queries, topk, ef_search,
        _nns.mutable_data(0), _distances.mutable_data(0), _found_cnt.mutable_data(0));
  }

 private:
  cuhnsw::CuHNSW obj_;
};

PYBIND11_PLUGIN(cuhnsw_bind) {
  py::module m("CuHNSWBind");

  py::class_<CuHNSWBind>(m, "CuHNSWBind")
  .def(py::init())
  .def("init", &CuHNSWBind::Init, py::arg("opt_path"))
  .def("set_data", &CuHNSWBind::SetData, py::arg("data"))
  .def("build_graph", &CuHNSWBind::BuildGraph)
  .def("set_random_levels", &CuHNSWBind::SetRandomLevels, py::arg("levels"))
  .def("save_index", &CuHNSWBind::SaveIndex, py::arg("fpath"))
  .def("load_index", &CuHNSWBind::LoadIndex, py::arg("fpath"))
  .def("search_knn", &CuHNSWBind::SearchGraph,
      py::arg("qdata"), py::arg("topk"), py::arg("ef_search"),
      py::arg("nns"), py::arg("distances"), py::arg("found"))
  .def("__repr__",
  [](const CuHNSWBind &a) {
    return "<CuHNSWBind>";
  }
  );
  return m.ptr();
}

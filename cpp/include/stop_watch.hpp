// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <time.h>
#include <utility>

class StopWatch {
 public:
  StopWatch() {
    clock_gettime(CLOCK_MONOTONIC, &beg_);
  }
  ~StopWatch() {}
  inline double CheckPoint() {
    clock_gettime(CLOCK_MONOTONIC, &end_);
    double ret = (end_.tv_sec - beg_.tv_sec) + (end_.tv_nsec - beg_.tv_nsec) / 1e9;
    std::swap(beg_, end_);
    return ret;
  }
 private:
  timespec beg_, end_;
};

// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include <cuda_fp16.h>

// experimental codes to use half precision
// not properly working yet..
// #define HALF_PRECISION 1

// #if __CUDA_ARCH__ < 530
//   #undef HALF_PRECISION
// #endif

#ifdef HALF_PRECISION
  typedef half cuda_scalar;
  #define mul(x, y) ( __hmul(x, y) )
  #define add(x, y) ( __hadd(x, y) )
  #define sub(x, y) ( __hsub(x, y) )
  #define gt(x, y) ( __hgt(x, y) )  // x > y
  #define ge(x, y) ( __hge(x, y) )  // x >= y
  #define lt(x, y) ( __hlt(x, y) )  // x < y
  #define le(x, y) ( __hle(x, y) )  // x <= y
  #define out_scalar(x) ( __half2float(x) )
  #define conversion(x) ( __float2half(x) )
#else
  typedef float cuda_scalar;
  #define mul(x, y) ( x * y )
  #define add(x, y) ( x + y )
  #define sub(x, y) ( x - y )
  #define gt(x, y) ( x > y )
  #define ge(x, y) ( x >= y )
  #define lt(x, y) ( x < y )
  #define le(x, y) ( x <= y )
  #define out_scalar(x) ( x )
  #define conversion(x) ( x )
#endif

#define WARP_SIZE 32

struct Neighbor {
  cuda_scalar distance;
  int nodeid;
  bool checked;
};

// to manage the compatibility with hnswlib
typedef unsigned int tableint;
typedef unsigned int sizeint;
typedef float scalar;
typedef size_t labeltype;

enum DIST_TYPE {
  DOT,
  L2,
};

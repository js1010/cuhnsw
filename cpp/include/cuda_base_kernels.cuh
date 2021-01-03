// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include <unistd.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <cooperative_groups.h>

#include <stdexcept>
#include <sstream>
#include <ctime>
#include <utility>

#include "types.hpp"

namespace cuhnsw {

// Error Checking utilities, checks status codes from cuda calls
// and throws exceptions on failure (which cython can proxy back to python)
#define CHECK_CUDA(code) { checkCuda((code), __FILE__, __LINE__); }
inline void checkCuda(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::stringstream err;
    err << "Cuda Error: " << cudaGetErrorString(code) << " (" << file << ":" << line << ")";
    throw std::runtime_error(err.str());
  }
}

inline const char* cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "Unknown";
}

#define CHECK_CUBLAS(code) { checkCublas((code), __FILE__, __LINE__); }
inline void checkCublas(cublasStatus_t code, const char * file, int line) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    std::stringstream err;
    err << "cublas error: " << cublasGetErrorString(code)
        << " (" << file << ":" << line << ")";
    throw std::runtime_error(err.str());
  }
}
} // namespace cuhnsw

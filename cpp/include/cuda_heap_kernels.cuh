// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include "cuda_base_kernels.cuh"

namespace cuhnsw {

// pop and push for heap
// reference: https://github.com/NVlabs/nvbio/blob/master/nvbio/basic/priority_queue_inline.h
__inline__ __device__
void PqPop(Neighbor* pq, int* size) {
  if (threadIdx.x != 0) return;
  if (*size == 0) return;
  (*size)--;
  if (*size == 0) return;
  cuda_scalar tail_dist = pq[*size].distance;
  int p = 0, r = 1;
  while (r < *size) {
    if (r < (*size) - 1 and gt(pq[r + 1].distance, pq[r].distance))
      r++;
    if (ge(tail_dist, pq[r].distance))
      break;
    pq[p] = pq[r];
    p = r;
    r = 2 * p + 1;
  }
  pq[p] = pq[*size];
}

__inline__ __device__
void PqPush(Neighbor* pq, int* size,
    float dist, int nodeid, bool check) {
  if (threadIdx.x != 0) return;
  int idx = *size;
  while (idx > 0) {
    int nidx = (idx + 1) / 2 - 1;
    if (ge(pq[nidx].distance, dist))
      break;
    pq[idx] = pq[nidx];
    idx = nidx;
  }
  pq[idx].distance = dist;
  pq[idx].nodeid = nodeid;
  pq[idx].checked = check;
  (*size)++;
}

} // namespace cuhnsw

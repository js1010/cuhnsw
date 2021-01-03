# Copyright (c) 2020 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,too-few-public-methods,no-member
import os
import json
import tempfile

import numpy as np

from cuhnsw import aux
from cuhnsw.cuhnsw_bind import CuHNSWBind

EPS = 1e-10
WARP_SIZE = 32

class CuHNSW:
  def __init__(self, opt=None):
    self.opt = aux.get_opt_as_proto(opt or {})

    self.opt.level_mult = self.opt.level_mult or 1 / np.log(self.opt.max_m)
    self.logger = aux.get_logger("cuhnsw", self.opt.py_log_level)
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    opt_content = json.dumps(aux.proto_to_dict(self.opt), indent=2)
    tmp.write(opt_content)
    tmp.close()
    self.logger.info("opt: %s", opt_content)
    self.data = None
    self.obj = CuHNSWBind()
    assert self.opt.block_dim <= WARP_SIZE ** 2 and \
      self.opt.block_dim % WARP_SIZE == 0, \
      f"invalid block dim ({self.opt.block_dim}, warp size: {WARP_SIZE})"
    assert self.obj.init(bytes(tmp.name, "utf8")), \
      f"failed to load {tmp.name}"
    os.remove(tmp.name)

  def set_data(self, data):
    self.data = data.copy()
    if self.opt.nrz:
      self.data /= np.linalg.norm(self.data, axis=1)[:, None]
    num_data, num_dims = self.data.shape
    self.logger.info("data shape: %d x %d", num_data, num_dims)
    self.obj.set_data(self.data)

  def build(self):
    self.set_random_levels()
    self.obj.build_graph()

  def set_random_levels(self):
    np.random.seed(self.opt.seed)
    num_data = self.data.shape[0]
    levels = np.random.uniform(size=num_data)
    levels = np.maximum(levels, EPS)
    levels = (-np.log(levels) * self.opt.level_mult).astype(np.int32)
    self.obj.set_random_levels(levels)

  def save_index(self, fpath):
    self.obj.save_index(fpath.encode("utf-8"))

  def load_index(self, fpath):
    self.obj.load_index(fpath.encode("utf-8"))

  def search_knn(self, qdata, topk, ef_search):
    ef_search = max(topk, ef_search)
    qdata = qdata.astype(np.float32)
    num_queries = qdata.shape[0]
    nns = np.empty(shape=(num_queries, topk), dtype=np.int32)
    distances = np.empty(shape=(num_queries, topk), dtype=np.float32)
    found_cnt = np.empty(shape=(num_queries,), dtype=np.int32)
    self.obj.search_knn(qdata, topk, ef_search,
                        nns, distances, found_cnt)
    return nns, distances, found_cnt

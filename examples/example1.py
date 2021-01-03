# Copyright (c) 2020 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,logging-format-truncated
import os
import time
import subprocess

import h5py
import fire
# import tqdm
import numpy as np

import hnswlib
from cuhnsw import aux, CuHNSW

LOGGER = aux.get_logger()

DATA_PATH = "glove-50-angular.hdf5"
CUHNSW_INDEX_PATH = "cuhnsw.index"
HNSWLIB_INDEX_PATH = "hnswlib.index"
TARGET_INDEX_PATH = "hnswlib.index"
DATA_URL = "http://ann-benchmarks.com/glove-50-angular.hdf5"
DIST_TYPE = "dot"
NRZ = DIST_TYPE == "dot"
OPT = { \
  "data_path": DATA_PATH,
  "c_log_level": 3,
  "ef_construction": 100,
  "hyper_threads": 20,
  "block_dim": 32,
  "nrz": NRZ,
  "reverse_cand": False,
  # "heuristic_coef": 0.0,
  "dist_type": DIST_TYPE, \
}


def download():
  if os.path.exists(DATA_PATH):
    return
  cmds = ["wget", DATA_URL, "-O", DATA_PATH + ".tmp"]
  cmds = " ".join(cmds)
  LOGGER.info("download data: %s", cmds)
  subprocess.call(cmds, shell=True)
  os.rename(DATA_PATH + ".tmp", DATA_PATH)


def run_cpu_inference(topk=100, ef_search=100,
                      target=TARGET_INDEX_PATH, evaluate=True):
  h5f = h5py.File(DATA_PATH, "r")
  num_data = h5f["train"].shape[0]
  queries = h5f["test"][:, :].astype(np.float32)
  neighbors = h5f["neighbors"][:, :topk].astype(np.int32)
  h5f.close()
  hl0 = hnswlib.Index(space="ip", dim=queries.shape[1])
  LOGGER.info("load %s by hnswlib", target)
  num_queries = queries.shape[0]
  hl0.load_index(target, max_elements=num_data)
  hl0.set_ef(ef_search)
  queries /= np.linalg.norm(queries, axis=1)[:, None]

  start = time.time()
  labels, _ = hl0.knn_query(queries, k=topk, num_threads=1)
  LOGGER.info("elapsed for processing %d queries computing top@%d: %.4e sec",
              num_queries, topk, time.time() - start)
  if evaluate:
    accs = []
    for _pred_nn, _gt_nn in zip(labels, neighbors):
      intersection = set(_pred_nn) & set(_gt_nn)
      acc = len(intersection) / float(topk)
      accs.append(acc)
    LOGGER.info("accuracy mean: %.4e, std: %.4e", np.mean(accs), np.std(accs))

def run_cpu_training(ef_const=150, num_threads=-1):
  h5f = h5py.File(DATA_PATH, "r")
  data = h5f["train"][:, :].astype(np.float32)
  h5f.close()
  hl0 = hnswlib.Index(space="ip", dim=data.shape[1])
  num_data = data.shape[0]
  data /= np.linalg.norm(data, axis=1)[:, None]
  hl0.init_index(max_elements=num_data, ef_construction=ef_const, M=12)
  LOGGER.info("add data to hnswlib")
  start = time.time()
  hl0.add_items(data, np.arange(num_data, dtype=np.int32),
                num_threads=num_threads)
  LOGGER.info("elapsed for adding %d items: %.4e sec",
              num_data, time.time() - start)
  hl0.save_index(HNSWLIB_INDEX_PATH)
  LOGGER.info("index saved to %s", HNSWLIB_INDEX_PATH)

def run_gpu_inference(topk=100, target=TARGET_INDEX_PATH,
                      ef_search=100, evaluate=True):
  ch0 = CuHNSW(OPT)
  LOGGER.info("load model from %s by cuhnsw", target)
  ch0.load_index(target)

  h5f = h5py.File(DATA_PATH, "r")
  queries = h5f["test"][:, :].astype(np.float32)
  neighbors = h5f["neighbors"][:, :topk].astype(np.int32)
  h5f.close()
  num_queries = queries.shape[0]
  queries /= np.linalg.norm(queries, axis=1)[:, None]

  start = time.time()
  pred_nn, _, _ = ch0.search_knn(queries, topk, ef_search)
  LOGGER.info("elapsed for inferencing %d queries of top@%d (ef_search: %d): "
              "%.4e sec", num_queries, topk, ef_search, time.time() - start)
  if evaluate:
    accs = []
    for _pred_nn, _gt_nn in zip(pred_nn, neighbors):
      intersection = set(_pred_nn) & set(_gt_nn)
      acc = len(intersection) / float(topk)
      accs.append(acc)
    LOGGER.info("accuracy mean: %.4e, std: %.4e", np.mean(accs), np.std(accs))

def run_gpu_training(ef_const=150, reverse_cand=False):
  OPT["reverse_cand"] = reverse_cand
  OPT["ef_construction"] = ef_const
  ch0 = CuHNSW(OPT)
  ch0.set_data()
  start = time.time()
  ch0.build_graph()
  LOGGER.info("elpased to build graph: %f sec", time.time() - start)
  ch0.save_index(CUHNSW_INDEX_PATH)



if __name__ == "__main__":
  fire.Fire()

# Copyright (c) 2020 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,logging-format-truncated
import os
from os.path import join as pjoin
import time
import subprocess

import h5py
import fire
# import tqdm
import numpy as np
import pandas as pd

import hnswlib
from cuhnsw import aux, CuHNSW

LOGGER = aux.get_logger()

NUM_DATA = 1183514
DATA_FILE = "glove-50-angular.hdf5"
DIST_TYPE = "dot"

# NUM_DATA = 1000000
# DATA_FILE = "sift-128-euclidean.hdf5"
# DIST_TYPE = "l2"

BARRIER_SIZE = 100
RES_DIR = "res"
INDEX_FILE = "hnswlib.index"
CUHNSW_INDEX_FILE = "cuhnsw.index"
HNSWLIB_INDEX_FILE = "hnswlib.index"
DATA_URL = f"http://ann-benchmarks.com/{DATA_FILE}"
HNSW_DIST_MAP = {"dot": "ip"}
HNSW_DIST = HNSW_DIST_MAP.get(DIST_TYPE, DIST_TYPE)
NRZ = DIST_TYPE == "dot"
OPT = { \
  "c_log_level": 2,
  "ef_construction": 150,
  "hyper_threads": 100,
  "block_dim": 32,
  "nrz": NRZ,
  "reverse_cand": False,
  "heuristic_coef": 0.0,
  "dist_type": DIST_TYPE, \
}


def download():
  if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)
  data_path = pjoin(RES_DIR, DATA_FILE)
  if os.path.exists(data_path):
    return
  cmds = ["wget", DATA_URL, "-O", data_path + ".tmp"]
  cmds = " ".join(cmds)
  LOGGER.info("download data: %s", cmds)
  subprocess.call(cmds, shell=True)
  os.rename(data_path + ".tmp", data_path)


def run_cpu_inference(topk=100, ef_search=300, index_file=INDEX_FILE,
                      evaluate=True, num_threads=-1):
  print("=" * BARRIER_SIZE)
  data_path = pjoin(RES_DIR, DATA_FILE)
  index_path = pjoin(RES_DIR, index_file)
  LOGGER.info("cpu inference on %s with index %s", data_path, index_path)
  h5f = h5py.File(data_path, "r")
  num_data = h5f["train"].shape[0]
  queries = h5f["test"][:, :].astype(np.float32)
  neighbors = h5f["neighbors"][:, :topk].astype(np.int32)
  h5f.close()
  hl0 = hnswlib.Index(space=HNSW_DIST, dim=queries.shape[1])
  LOGGER.info("load %s by hnswlib", index_path)
  num_queries = queries.shape[0]
  hl0.load_index(index_path, max_elements=num_data)
  hl0.set_ef(ef_search)
  if NRZ:
    queries /= np.linalg.norm(queries, axis=1)[:, None]

  start = time.time()
  labels, _ = hl0.knn_query(queries, k=topk, num_threads=num_threads)
  el0 = time.time() - start
  LOGGER.info("elapsed for processing %d queries computing top@%d: %.4e sec",
              num_queries, topk, el0)
  if evaluate:
    accs = []
    for _pred_nn, _gt_nn in zip(labels, neighbors):
      intersection = set(_pred_nn) & set(_gt_nn)
      acc = len(intersection) / float(topk)
      accs.append(acc)
    LOGGER.info("accuracy mean: %.4e, std: %.4e", np.mean(accs), np.std(accs))
    return el0, np.mean(accs)
  return el0

def run_cpu_inference_large(topk=100, index_file=INDEX_FILE, ef_search=300,
                            num_queries=1000000, num_dims=50, num_threads=-1):
  print("=" * BARRIER_SIZE)
  index_path = pjoin(RES_DIR, index_file)
  data_path = pjoin(RES_DIR, DATA_FILE)
  LOGGER.info("cpu inference on %s with index %s", data_path, index_path)

  queries = np.random.normal(size=(num_queries, num_dims)).astype(np.float32)
  queries /= np.linalg.norm(queries, axis=1)[:, None]

  hl0 = hnswlib.Index(space=HNSW_DIST, dim=queries.shape[1])
  LOGGER.info("load %s by hnswlib", index_path)
  hl0.load_index(index_path, max_elements=NUM_DATA)
  hl0.set_ef(ef_search)
  queries /= np.linalg.norm(queries, axis=1)[:, None]

  start = time.time()
  _, _ = hl0.knn_query(queries, k=topk, num_threads=num_threads)
  el0 = time.time() - start
  LOGGER.info("elapsed for inferencing %d queries of top@%d (ef_search: %d): "
              "%.4e sec", num_queries, topk, ef_search, el0)
  return el0

def run_cpu_training(ef_const=150, num_threads=-1):
  print("=" * BARRIER_SIZE)
  data_path = pjoin(RES_DIR, DATA_FILE)
  LOGGER.info("cpu training on %s with ef const %d, num_threads: %d",
              data_path, ef_const, num_threads)
  h5f = h5py.File(data_path, "r")
  data = h5f["train"][:, :].astype(np.float32)
  h5f.close()
  hl0 = hnswlib.Index(space=HNSW_DIST, dim=data.shape[1])
  num_data = data.shape[0]
  data /= np.linalg.norm(data, axis=1)[:, None]
  hl0.init_index(max_elements=num_data, ef_construction=ef_const, M=12)
  LOGGER.info("add data to hnswlib")
  start = time.time()
  hl0.add_items(data, np.arange(num_data, dtype=np.int32),
                num_threads=num_threads)
  el0 = time.time() - start
  LOGGER.info("elapsed for adding %d items: %.4e sec", num_data, el0)
  index_path = pjoin(RES_DIR, HNSWLIB_INDEX_FILE)
  hl0.save_index(index_path)
  LOGGER.info("index saved to %s", index_path)
  return el0

def run_gpu_inference(topk=100, index_file=INDEX_FILE, ef_search=300):
  print("=" * BARRIER_SIZE)
  data_path = pjoin(RES_DIR, DATA_FILE)
  index_path = pjoin(RES_DIR, index_file)
  LOGGER.info("gpu inference on %s with index %s", data_path, index_path)
  ch0 = CuHNSW(OPT)
  LOGGER.info("load model from %s by cuhnsw", index_path)
  ch0.load_index(index_path)

  h5f = h5py.File(data_path, "r")
  queries = h5f["test"][:, :].astype(np.float32)
  neighbors = h5f["neighbors"][:, :topk].astype(np.int32)
  h5f.close()
  num_queries = queries.shape[0]
  if NRZ:
    queries /= np.linalg.norm(queries, axis=1)[:, None]

  start = time.time()
  pred_nn, _, _ = ch0.search_knn(queries, topk, ef_search)
  el0 = time.time() - start
  LOGGER.info("elapsed for inferencing %d queries of top@%d (ef_search: %d): "
              "%.4e sec", num_queries, topk, ef_search, el0)
  accs = []
  for _pred_nn, _gt_nn in zip(pred_nn, neighbors):
    intersection = set(_pred_nn) & set(_gt_nn)
    acc = len(intersection) / float(topk)
    accs.append(acc)
  LOGGER.info("accuracy mean: %.4e, std: %.4e", np.mean(accs), np.std(accs))
  return el0, np.mean(accs)

def run_gpu_inference2(topk=5, index_file="cuhnsw.index", ef_search=300):
  print("=" * BARRIER_SIZE)
  data_path = pjoin(RES_DIR, DATA_FILE)
  index_path = pjoin(RES_DIR, index_file)
  LOGGER.info("gpu inference on %s with index %s", data_path, index_path)
  ch0 = CuHNSW(OPT)
  LOGGER.info("load model from %s by cuhnsw", index_path)
  ch0.load_index(index_path)

  h5f = h5py.File(data_path, "r")
  data = h5f["train"][:, :].astype(np.float32)
  queries = h5f["test"][:5, :].astype(np.float32)
  h5f.close()
  if NRZ:
    data /= np.linalg.norm(data, axis=1)[:, None]

  nns, distances, found_cnt = ch0.search_knn(queries[:5], topk, ef_search)
  for idx, (nn0, distance, cnt) in \
      enumerate(zip(nns, distances, found_cnt)):
    print("=" * BARRIER_SIZE)
    print(f"query {idx + 1}")
    print("-" * BARRIER_SIZE)
    for _idx, (_nn, _dist) in enumerate(zip(nn0[:cnt], distance[:cnt])):
      if DIST_TYPE == "l2":
        real_dist = np.linalg.norm(data[_nn] - queries[idx])
        _dist = np.sqrt(_dist)
      elif DIST_TYPE == "dot":
        real_dist = data[_nn].dot(queries[idx])
      print(f"rank {_idx + 1}. neighbor: {_nn}, dist by lib: {_dist}, "
            f"actual dist: {real_dist}")


def run_gpu_inference_large(topk=100, index_file=INDEX_FILE, ef_search=300,
                            num_queries=1000000, num_dims=50):
  print("=" * BARRIER_SIZE)
  index_path = pjoin(RES_DIR, index_file)
  data_path = pjoin(RES_DIR, DATA_FILE)
  LOGGER.info("gpu inference on %s with index %s", data_path, index_path)
  ch0 = CuHNSW(OPT)
  LOGGER.info("load model from %s by cuhnsw", index_path)
  ch0.load_index(index_path)

  queries = np.random.normal(size=(num_queries, num_dims)).astype(np.float32)
  num_queries = queries.shape[0]
  if NRZ:
    queries /= np.linalg.norm(queries, axis=1)[:, None]

  start = time.time()
  _, _, _ = ch0.search_knn(queries, topk, ef_search)
  el0 = time.time() - start
  LOGGER.info("elapsed for inferencing %d queries of top@%d (ef_search: %d): "
              "%.4e sec", num_queries, topk, ef_search, el0)
  return el0

def run_gpu_training(ef_const=150):
  print("=" * BARRIER_SIZE)
  data_path = pjoin(RES_DIR, DATA_FILE)
  LOGGER.info("gpu training on %s with ef const %d", data_path, ef_const)
  OPT["ef_construction"] = ef_const
  ch0 = CuHNSW(OPT)
  h5f = h5py.File(data_path, "r")
  data = h5f["train"][:, :].astype(np.float32)
  h5f.close()
  ch0.set_data(data)
  start = time.time()
  ch0.build()
  el0 = time.time() - start
  LOGGER.info("elpased time to build by cuhnsw: %.4e sec", el0)
  index_path = pjoin(RES_DIR, CUHNSW_INDEX_FILE)
  ch0.save_index(index_path)
  return el0

def measure_build_performance():
  build_time = {"attr": "build time"}
  build_quality = {"attr": "build quality"}
  build_time["gpu"] = run_gpu_training(ef_const=110)
  _, build_quality["gpu"] = run_gpu_inference(index_file="cuhnsw.index")
  for i in [1, 2, 4, 8]:
    build_time[f"{i} cpu"] = run_cpu_training(ef_const=150, num_threads=i)
    _, build_quality[f"{i} cpu"] = run_cpu_inference(index_file="hnswlib.index")
  columns = [f"{i} cpu" for i in [1, 2, 4, 8]] + ["gpu"]
  df0 = pd.DataFrame([build_time, build_quality])
  df0.set_index("attr", inplace=True)
  print(df0[columns].to_markdown())

def measure_search_performance():
  search_time = {"attr": "search time"}
  search_time["gpu"] = run_gpu_inference_large(index_file="cuhnsw.index")
  for i in [1, 2, 4, 8]:
    search_time[f"{i} cpu"] = run_cpu_inference_large(
      index_file="cuhnsw.index", num_threads=i)
  columns = [f"{i} cpu" for i in [1, 2, 4, 8]] + ["gpu"]
  df0 = pd.DataFrame([search_time])
  df0.set_index("attr", inplace=True)
  print(df0[columns].to_markdown())


def run_experiments():
  measure_build_performance()
  measure_search_performance()


if __name__ == "__main__":
  fire.Fire()

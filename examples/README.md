### How to run example code

1. first, it is good to know about python-fire in https://github.com/google/python-fire, if you haven't heard yet.
2. download data

```
# python example1.py download
[INFO    ] 2021-01-03 22:10:29 [example1.py] [download:47]download data: wget http://ann-benchmarks.com/glove-50-angular.hdf5 -O glove-50-angular.hdf5.tmp
--2021-01-03 22:10:29--  http://ann-benchmarks.com/glove-50-angular.hdf5
Resolving ann-benchmarks.com (ann-benchmarks.com)... 52.216.204.210
Connecting to ann-benchmarks.com (ann-benchmarks.com)|52.216.204.210|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 246711088 (235M) [binary/octet-stream]
Saving to: 'glove-50-angular.hdf5.tmp'

glove-50-angular.hdf5.tmp        100%[========================================================>] 235.28M  7.54MB/s    in 32s

2021-01-03 22:11:01 (7.46 MB/s) - 'glove-50-angular.hdf5.tmp' saved [246711088/246711088]
```

3. run gpu training


```
# python example1.py run_gpu_training
[INFO    ] 2021-01-03 22:11:33 [pyhnsw.py] [__init__:30]opt: {
  "data_path": "glove-50-angular.hdf5",
  "c_log_level": 2,
  "ef_construction": 150,
  "level_mult": 0.40242960438184466,
  "hyper_threads": 20.0,
  "block_dim": 32,
  "dist_type": "dot",
  "nrz": true,
  "reverse_cand": false,
  "seed": 777,
  "py_log_level": 2,
  "max_m": 12,
  "max_m0": 24,
  "save_remains": false,
  "visited_table_size": 0,
  "visited_list_size": 8192,
  "heuristic_coef": 0.25
}
[info    ] 2021-01-03 22:11:33 [cuhnsw_base.cu:59] cuda device info, major: 7, minor: 5, multi processors: 40, cores: 2560
[INFO    ] 2021-01-03 22:11:34 [pyhnsw.py] [set_data:45]data shape: 1183514 x 50
[INFO    ] 2021-01-03 22:12:14 [example1.py] [run_gpu_training:130]elpased time to build by cuhnsw: 4.0361e+01 sec
```

4. check the saved index (filename: cuhnsw.index)


```
# ll
total 593M
drwxr-xr-x  2 root root 4.0K Jan  3 22:11 .
drwxr-xr-x 11 root root 4.0K Jan  3 22:02 ..
-rw-r--r--  1 root root   15 Jan  3 19:50 .gitignore
-rw-r--r--  1 root root  967 Jan  3 22:11 README.md
-rw-r--r--  1 root root 358M Jan  3 22:12 cuhnsw.index
-rw-r--r--  1 root root 4.2K Jan  3 22:07 example1.py
-rw-r--r--  1 root root 236M Jun 29  2018 glove-50-angular.hdf5
```

5. search the nearest neighbors loading the file in cuhnsw (GPU)

```
# python example1.py run_gpu_inference --target=cuhnsw.index --topk=10
[INFO    ] 2021-01-03 22:25:44 [pyhnsw.py] [__init__:30]opt: {
  "data_path": "glove-50-angular.hdf5",
  "c_log_level": 2,
  "ef_construction": 100,
  "level_mult": 0.40242960438184466,
  "hyper_threads": 20.0,
  "block_dim": 32,
  "dist_type": "dot",
  "nrz": true,
  "reverse_cand": false,
  "seed": 777,
  "py_log_level": 2,
  "max_m": 12,
  "max_m0": 24,
  "save_remains": false,
  "visited_table_size": 0,
  "visited_list_size": 8192,
  "heuristic_coef": 0.25
}
[info    ] 2021-01-03 22:25:44 [cuhnsw_base.cu:59] cuda device info, major: 7, minor: 5, multi processors: 40, cores: 2560
[INFO    ] 2021-01-03 22:25:44 [example1.py] [run_gpu_inference:98]load model from cuhnsw.index by cuhnsw
[INFO    ] 2021-01-03 22:25:46 [example1.py] [run_gpu_inference:111]elapsed for inferencing 10000 queries of top@10 (ef_search: 300): 1.0471e+00 sec
[INFO    ] 2021-01-03 22:25:46 [example1.py] [run_gpu_inference:118]accuracy mean: 9.3027e-01, std: 1.3084e-01
```

6. you can also search the nearest neighbor by hnswlib (CPU)

```
# python example1.py run_cpu_inference --target=cuhnsw.index --topk=10
[INFO    ] 2021-01-03 22:25:11 [example1.py] [run_cpu_inference:60]load cuhnsw.index by hnswlib
[INFO    ] 2021-01-03 22:25:18 [example1.py] [run_cpu_inference:69]elapsed for processing 10000 queries computing top@10: 4.7242e+00 sec
[INFO    ] 2021-01-03 22:25:18 [example1.py] [run_cpu_inference:76]accuracy mean: 9.3027e-01, std: 1.3084e-01
```

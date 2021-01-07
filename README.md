### Introduction

This project is to speed up HNSW algorithm by CUDA. I expect that anyone who will be interested in this project might be already familiar with the following paper and the open source project. If not, I strongly recommend that you check them first.

- hnsw paper: https://arxiv.org/pdf/1603.09320.pdf (2016)
- hnsw implementation (cpu only) by the author of hnsw (Yury Markov): https://github.com/nmslib/hnswlib
- Approximate Nearest Neighbor (ANN) Benchmark Site: http://ann-benchmarks.com/

I also adapted some ideas from the following project.

- n2 (alternative hnsw cpu implementation project): https://github.com/kakao/n2

By brief survey, I found there are several papers and projects to suggest to speed up ANN algorithms by GPU.

- papers or projects related to using GPU for ANN
  - paper (2020): http://research.baidu.com/Public/uploads/5f5c37aa9c37c.pdf
  - paper (2017): https://arxiv.org/pdf/1702.05911.pdf
  - slides (2020): https://wangzwhu.github.io/home/file/acmmm-t-part3-ann.pdf
  - project (2017): https://github.com/facebookresearch/faiss
  
I started this project because I was originally interested in both CUDA programming and ANN algorithms. I release this project because it achieved meaningful performance and hope to develop further by community participation. 

Literally, this package is implemented to build HNSW graphs using GPU, and to approximate nearest neighbor search through the built graphs, and the format of the model file is compatible with hnswlib. In other words, you can build a HNSW graph from this package, then save it and load it from hnswlib for search, and vice versa.


### How to install


```shell
# clone repo and submodules
git clone git@github.com:js1010/cuhnsw.git && cd cuhnsw && git submodule update --init

# install requirements
pip install -r requirements.txt

# generate proto
python -m grpc_tools.protoc --python_out cuhnsw/ --proto_path cuhnsw/proto/ config.proto

# install
python setup.py install
```

### How to use

- `examples/example1.py` and `examples/README.md` will be very helpful to understand the usage.
- build and save model

```python
import h5py
from cuhnsw import CuHNSW


h5f = h5py.File("glove-50-angular.hdf5", "r")
data = h5f["train"][:, :].astype(np.float32)
h5f.close()
ch0 = CuHNSW(opt={})
ch0.set_data(data)
ch0.build()
ch0.save_index("cuhnsw.index")
```

- load model and search

```python
import h5py
from cuhnsw import CuHNSW

h5f = h5py.File("glove-50-angular.hdf5", "r")
data = h5f["test"][:, :].astype(np.float32)
h5f.close()
ch0 = CuHNSW(opt={})
ch0.load_index("cuhnsw.index")
nns, distances, found_cnt = ch0.search_knn(data, topk=10, ef_search=300)
```

- Option parameters (see `cuhnsw/proto/config.proto`)
  - `seed`: numpy random seed (used in random levels)
  - `c_log_level`: log level in cpp logging (spdlog)
  - `py_log_level`: log level in python logging
  - `max_m`: maximum number of links in layers higher than ground layer
  - `max_m0`: maximum number of links in the ground layer
  - `level_mult`: multiplier to draw levels of each element (defualt: 0 => setted as `1 / log(max_m0)` in initialization as recommended in hnsw paper)
  - `save_remains`: link to remained candidates in SearchHeuristic (adapted from n2)
  - `heuristic_coff`: select some closest candidates by default (also adapted from n2)
  - `hyper_threads`: set the number of gpu blocks as the total number of concurrent cores exceeds the physical number of cores
  - `block_dim`: block dimension (should be smaller than 32^2=1024 and should be the multiple of 32)
  - `nrz`: normalize data vector if True
  - `visited_table_size`: size of table to store the visited nodes in each search
  - `visited_list_size`: size of list to store the visited nodes in each search (useful to reset table after each search)
  - `reverse_cand`: select the candidate with the furthest distance if True (it makes the build slower but achieves better quality)
  - `dist_type`: euclidean distance if "l2" and inner product distaance if "dot"

### Performance

- tl;dr
  - cuhnsw achieved the same build quality by 8~9 times faster build time than hnswlib with 8 vcpus on certain data and parameter setup
  - cuhnsw achieved the same search quality by 3 times faster search time than hnswlib with 8 vcpus instance on certain data and parameter setup
- Note1: HNSW search algorithm can be verified by exact match since it is deterministic. 
  - I verified it with hnswlib, in other words, cuhnsw search and hnswlib search returns exactly same results by loading the same model file and the same queries and the same ef search.
- Note2: GPU search has the advantage over CPU search only when it comes to the `Batch` search (i.e. processing large number of queries at once.) 
- [AWS P3 2xlarge instance](https://aws.amazon.com/ec2/instance-types/p3/) is used to the experiment. (One Tesla V100 GPU with 8 vcpus)
- results can be reproduced by running `example/example1.py`.
- build time / quality results on glove-50-angular
  - used `ef_construction`=150 for hnswlib and `ef_construction=110` for cuhnsw to achieve the same build quality
  - build quality is measured by the accuracy by the same search parameter (`ef_search`=300)

| attr          |     1 vcpu |     2 vcpu |    4 vcpu |    8 vcpu |       gpu |
|:--------------|-----------:|-----------:|----------:|----------:|----------:|
| build time    | 343.909    | 179.836    | 89.7936   | 70.5476   | 8.2847    |
| build quality |   0.863193 |   0.863301 |  0.863238 |  0.863165 |  0.865471 |

- search time comparison on glove-50-angular
  - search time on 100k random queries
  - search `quality` is guaranteed to the same (exact match)

| attr        |  1 vcpu |  2 vcpu |  4 vcpu |  8 vcpu |     gpu |
|:------------|--------:|--------:|--------:|--------:|--------:|
| search time | 52.3024 | 26.5086 | 13.9146 | 10.8525 | 3.07964 |

- the reason why the parallel efficiency significantly drops from 4 vcpu to 8 vcpu might be hyper threading (there might be only 4 "physical" cores in this instance).

### Thoughts on Future Task

- Considering the cost of GPU and CPU, it seems impractical yet (currently one Tesla V100 device is equivalently fast as one standard cpu server (24-56 vcores) but the cost of Tesla V100 device is quite more expensive). Therefore, there seems to be a lot to do in the future.
- The word in the parentheses shows the expected level of difficulty for each task

1. **upload package to pypi (easy)**: the task itself is very easy but will not worth it unless this project is successful.
2. **implement parallel compilation using bazel or cmake (easy-medium)**: bazel is more preferable. compilation time is a little bit painful.
3. **achieve meaningful speed-up by using half-precision operation (medium)**: I experimented it, but only got around 10 % improvement. I am not sure if I have used the half-precision feature appropriately.
4. **support multi-device (very hard)**: it only supports single-device (gpu) yet since the graph should be shared across all the building threads.

- contribution is always welcome

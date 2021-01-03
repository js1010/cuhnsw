### Introductioon

This project is to speed up HNSW algorithm by CUDA. I expect that anyone who will be interested in this project might be already familiar with the following paper and the open source project. If not, I strongly recommend that you check them first.

- hnsw paper: https://arxiv.org/pdf/1603.09320.pdf (2016)
- hnsw implementation (cpu only) by the author of hnsw (Yury Markov): https://github.com/nmslib/hnswlib
- Approximate Nearest Neighbor (ANN) Benchmark Site: http://ann-benchmarks.com/

I also adapted some ideas from the following project.

- n2 (alternative hnsw cpu implementation project): https://github.com/kakao/n2

By brief survey, I found there are several papers to suggest to speed up ANN algorithms by GPU, but could not find single proper open sourced implementation yet.

- papers related to using GPU for ANN
  - http://research.baidu.com/Public/uploads/5f5c37aa9c37c.pdf (2020)
  - https://arxiv.org/pdf/1702.05911.pdf (2017)

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

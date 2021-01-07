### How to run example code

0. install requirements

```shell
# install python requirements
pip install -r requirements.txt

# install wget to download data (in ubuntu)
apt install wget
```

1. first, it is good to know about python-fire in https://github.com/google/python-fire, if you haven't heard yet.

2. download data

```shell
python example1.py download
```

3. run gpu training


```shell
python example1.py run_gpu_training
```

4. check the saved index (filename: `cuhnsw.index`)



5. search the nearest neighbors loading the file in cuhnsw (GPU)

```shell
python example1.py run_gpu_inference --index-file=cuhnsw.index --topk=10
```

6. you can also search the nearest neighbor by hnswlib (CPU)

```shell
python example1.py run_cpu_inference --index-file=cuhnsw.index --topk=10
```

7. reproduce the experimental results shown in README.md on the root directory

```shell
python example1.py run_experiments
```

### Installation

```
# install requirements
pip install -r requirements.txt
# generate proto
python -m grpc_tools.protoc --python_out cuhnsw/ --proto_path cuhnsw/proto/ config.proto
# install
python setup.py install
```

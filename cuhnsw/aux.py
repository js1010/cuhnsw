# Copyright (c) 2020 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import re
import json
import logging
import logging.handlers

import jsmin
from google.protobuf.json_format import Parse, MessageToDict
from cuhnsw.config_pb2 import ConfigProto

# get_logger and Option refer to
# https://github.com/kakao/buffalo/blob/
# 5f571c2c7d8227e6625c6e538da929e4db11b66d/buffalo/misc/aux.py
def get_logger(name=__file__, level=2):
  if level == 1:
    level = logging.WARNING
  elif level == 2:
    level = logging.INFO
  elif level == 3:
    level = logging.DEBUG
  logger = logging.getLogger(name)
  if logger.handlers:
    return logger
  logger.setLevel(level)
  sh0 = logging.StreamHandler()
  sh0.setLevel(level)
  formatter = logging.Formatter('[%(levelname)-8s] %(asctime)s '
                                '[%(filename)s] [%(funcName)s:%(lineno)d]'
                                '%(message)s', '%Y-%m-%d %H:%M:%S')
  sh0.setFormatter(formatter)
  logger.addHandler(sh0)
  return logger

# This function helps you to read non-standard json strings.
# - Handles json string with c++ style inline comments
# - Handles json string with trailing commas.
def load_json_string(cont):
  # (1) Removes comment.
  #     Refer to https://plus.google.com/+DouglasCrockfordEsq/posts/RK8qyGVaGSr
  cont = jsmin.jsmin(cont)

  # (2) Removes trailing comma.
  cont = re.sub(",[ \t\r\n]*}", "}", cont)
  cont = re.sub(",[ \t\r\n]*" + r"\]", "]", cont)

  return json.loads(cont)


# function read json file from filename
def load_json_file(fname):
  with open(fname, "r") as fin:
    ret = load_json_string(fin.read())
  return ret

# use protobuf to restrict field and types
def get_opt_as_proto(raw, proto_type=ConfigProto):
  proto = proto_type()
  # convert raw to proto
  Parse(json.dumps(Option(raw)), proto)
  err = []
  assert proto.IsInitialized(err), \
    f"some required fields are missing in proto {err}\n {proto}"
  return proto

def proto_to_dict(proto):
  return MessageToDict(proto, \
    including_default_value_fields=True, \
    preserving_proto_field_name=True)

def copy_proto(proto):
  newproto = type(proto)()
  Parse(json.dumps(proto_to_dict(proto)), newproto)
  return newproto

class Option(dict):
  def __init__(self, *args, **kwargs):
    args = [arg if isinstance(arg, dict)
            else load_json_file(arg) for arg in args]
    super().__init__(*args, **kwargs)
    for arg in args:
      if isinstance(arg, dict):
        for k, val in arg.items():
          if isinstance(val, dict):
            self[k] = Option(val)
          else:
            self[k] = val
    if kwargs:
      for k, val in kwargs.items():
        if isinstance(val, dict):
          self[k] = Option(val)
        else:
          self[k] = val

  def __getattr__(self, attr):
    return self.get(attr)

  def __setattr__(self, key, value):
    self.__setitem__(key, value)

  def __setitem__(self, key, value):
    super().__setitem__(key, value)
    self.__dict__.update({key: value})

  def __delattr__(self, item):
    self.__delitem__(item)

  def __delitem__(self, key):
    super().__delitem__(key)
    del self.__dict__[key]

  def __getstate__(self):
    return vars(self)

  def __setstate__(self, state):
    vars(self).update(state)

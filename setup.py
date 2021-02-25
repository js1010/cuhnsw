# Copyright (c) 2020 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=fixme,too-few-public-methods
# reference: https://github.com/kakao/buffalo/blob/
# 5f571c2c7d8227e6625c6e538da929e4db11b66d/setup.py
"""cuhnsw
"""
import os
import sys
import glob
import pathlib
import platform
import sysconfig
import subprocess
from setuptools import setup, Extension

import pybind11
import numpy as np
from cuda_setup import CUDA, BUILDEXT


DOCLINES = __doc__.split("\n")

# TODO: Python3 Support
if sys.version_info[:3] < (3, 6):
  raise RuntimeError("Python version 3.6 or later required.")

assert platform.system() == 'Linux'  # TODO: MacOS


MAJOR = 0
MINOR = 0
MICRO = 8
RELEASE = True
STAGE = {True: '', False: 'b'}.get(RELEASE)
VERSION = f'{MAJOR}.{MINOR}.{MICRO}{STAGE}'
STATUS = {False: 'Development Status :: 4 - Beta',
          True: 'Development Status :: 5 - Production/Stable'}

CLASSIFIERS = """{status}
Programming Language :: C++
Programming Language :: Python :: 3.6
Operating System :: POSIX :: Linux
Operating System :: Unix
Operating System :: MacOS
License :: OSI Approved :: Apache Software License""".format( \
  status=STATUS.get(RELEASE))
CLIB_DIR = os.path.join(sysconfig.get_path('purelib'), 'cuhnsw')
LIBRARY_DIRS = [CLIB_DIR]

with open("requirements.txt", "r") as fin:
  INSTALL_REQUIRES = [line.strip() for line in fin]

def get_extend_compile_flags():
  flags = ['-march=native']
  return flags


class CMakeExtension(Extension):
  extension_type = 'cmake'

  def __init__(self, name):
    super().__init__(name, sources=[])


extend_compile_flags = get_extend_compile_flags()
extra_compile_args = ['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + \
  extend_compile_flags
csrcs = glob.glob("cpp/src/*.cu") + glob.glob("cpp/src/*.cc")
extensions = [
  # CMakeExtension(name="cuhnsw"),
  Extension("cuhnsw.cuhnsw_bind",
            sources= csrcs + [ \
              "cuhnsw/bindings.cc",
              "3rd/json11/json11.cpp"],
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=["-fopenmp"],
            library_dirs=[CUDA['lib64']],
            libraries=['cudart', 'curand'],
            extra_objects=[],
            include_dirs=[ \
              "cpp/include/", np.get_include(),
              pybind11.get_include(), pybind11.get_include(True),
              CUDA['include'], "3rd/json11", "3rd/spdlog/include"])
]


# Return the git revision as a string
def git_version():
  def _minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ['SYSTEMROOT', 'PATH']:
      val = os.environ.get(k)
      if val is not None:
        env[k] = val
    out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env). \
      communicate()[0]
    return out

  try:
    out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
    git_revision = out.strip().decode('ascii')
  except OSError:
    git_revision = "Unknown"

  return git_revision


def write_version_py(filename='cuhnsw/version.py'):
  cnt = """
short_version = '%(version)s'
git_revision = '%(git_revision)s'
"""
  git_revision = git_version()
  with open(filename, 'w') as fout:
    fout.write(cnt % {'version': VERSION,
              'git_revision': git_revision})


class BuildExtension(BUILDEXT):
  def run(self):
    for ext in self.extensions:
      print(ext.name)
      if hasattr(ext, 'extension_type') and ext.extension_type == 'cmake':
        self.cmake()
    super().run()

  def cmake(self):
    cwd = pathlib.Path().absolute()

    build_temp = pathlib.Path(self.build_temp)
    build_temp.mkdir(parents=True, exist_ok=True)

    build_type = 'Debug' if self.debug else 'Release'

    cmake_args = [
      '-DCMAKE_BUILD_TYPE=' + build_type,
      '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + CLIB_DIR,
    ]

    build_args = []

    os.chdir(str(build_temp))
    self.spawn(['cmake', str(cwd)] + cmake_args)
    if not self.dry_run:
      self.spawn(['cmake', '--build', '.'] + build_args)
    os.chdir(str(cwd))


def setup_package():
  write_version_py()
  cmdclass = {
    'build_ext': BuildExtension
  }

  metadata = dict(
    name='cuhnsw',
    maintainer="Jisang Yoon",
    maintainer_email="vjs10101v@gmail.com",
    author="Jisang Yoon",
    author_email="vjs10101v@gmail.com",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    url="https://github.com/js1010/cuhnsw",
    download_url="https://github.com/js1010/cuhnsw/releases",
    include_package_data=False,
    license='Apache2',
    packages=['cuhnsw/'],
    cmdclass=cmdclass,
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms=['Linux', 'Mac OSX', 'Unix'],
    ext_modules=extensions,
    install_requires=INSTALL_REQUIRES,
    entry_points={
      'console_scripts': [
      ]
    },
    python_requires='>=3.6',
  )

  metadata['version'] = VERSION
  setup(**metadata)


if __name__ == '__main__':
  setup_package()

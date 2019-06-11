# Installation
Parallax runs under Linux with Python 2.7; we haven't yet tested Parallax on other platforms and 3.3+.
Parallax depends on a modified version of TensorFlow 1.6/1.11 and horovod 0.11.2 in parallax repository as submodules. *Each of these frameworks needs to be built and installed from source, which is explained in further detail below*. Parallax itself also requires installing from sources, and below explains the installation process step by step. We plan to provide binary files in the near future.

First, clone the parallax repository on your linux machine:
```shell
$ git clone --recurse-submodules https://github.com/snuspl/parallax.git
```
We recommend installing using Virtualenv and pip.

Install Python, pip, and Virtualenv:
```shell
$ sudo apt-get install python-pip python-dev python-virtualenv
```

Create a Virtualenv environment in the directory `parallax_venv`(specify whichever name you prefer), and then activate it.
```shell
$ virtualenv parallax_venv
$ source parallax_venv/bin/activate
```

## Install TensorFlow
TensorFlow requires [Bazel](https://docs.bazel.build/versions/master/install.html) to build a binary file. (See [TF install](https://www.tensorflow.org/install/install_sources) for more instructions on how to build TensorFlow from source.) TensorFlow can be built CPU-only but Parallax needs TensorFlow with GPU support using [CUDA Toolkit 9.0 or 10.0](https://developer.nvidia.com/cuda-zone) and [CuDNN SDK v7](https://developer.nvidia.com/cudnn). To install TensorFlow with GPU support, follow the commands below.

```shell
$ cd parallax/tensorflow
$ git checkout r1.11 (optional for TensorFlow v1.11)
$ pip install numpy
$ ./configure
  (Configurations related to cuda should be turned on to use GPUs)
  (verbs: ibverbs RDMA)
  (gdr: GPU Direct (only for GPUs with GDR support))
$ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package {target_directory}
$ pip install {target_directory}/tensorflow-*.whl
```


## Install Horovod
To install horovod, [Open MPI](https://www.open-mpi.org/faq/?category=building#easy-build) and [NCCL](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html) are required as MPI implementations. To install OpenMPI, `--with-cuda` flag should be in the configure line, and you can also add `--with-verbs` to use ibverbs.
We tested on openmpi-3.0.0, NCCL 2.1.15(for cuda9.0) and NCCL 2.3.5(for cuda10.0).
```shell
$ cd ../horovod
$ python setup.py sdist
$ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITHOUT_PYTORCH=True HOROVOD_WITHOUT_MXNET=True pip install --no-cache-dir dist/horovod-*.tar.gz
```

## Install Parallax
Parallax also uses [Bazel](https://docs.bazel.build/versions/master/install.html) for installation.
```shell
$ cd ../parallax # parallax directory
$ bazel build //parallax/util:build_pip_package
$ bazel-bin/parallax/util/build_pip_package {target_directory}
$ pip install {target_directory}/parallax-*.whl

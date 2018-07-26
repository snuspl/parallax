# Parallax Quick Start
Parallax mainly focuses on automatic parallelization of deep learning model graphs for a single device(CPU or GPU) with data parallelism. For automatic parallelization, Parallax requires some basic elements; resource information of the distributed environment, a single device graph, a data partitioning logic for data parallelism, a training method (asynchronous/synchronous), and a `run` function to invoke the computation of the distributed version of the graph.

## Resource Information
Parallax supports both communication methods of using Parameter Server(PS) or MPI implementation. Parallax automatically transforms a single device graph into a PS or MPI style graph, then compares their performance to choose the faster one. This process requires `master`, `ps`, and `worker` tasks. `master` is the center of communication with other processes to compare the throughput of PS and MPI. `ps` is only used for Parameter Server architecture for storing subsets of model parameter values in memory. `worker` is responsible for computations with a disjoint subset of input data. We assume a homogeneous environment where the number of GPUs per worker are equal.

Parallax receives resource information as a file with the IP address and GPU ids(optional), as shown below. If the GPU ids are not served, Parallax detects all the GPUs in the host. Parallax runs a `master` task at the first host and runs `ps` tasks(for Parameter Server architecture) in all the hosts in the resource information file.
Note that MPI creates a worker for each GPU device, so the number of workers are the same as the number of GPUs even if workers are set with multiple GPUs. Parallax assumes that all hosts are accessible by [ssh without password](http://www.linuxproblem.org/art_9.html).
```shell
12.34.56.789: 0,1,2,3,4,5
12.34.56.780: 1,3,4
12.34.56.781
```

## Single Device Graph
Parallax receives any complete TensorFlow graph that can run on a single device.

Below code is an example of a Parallax(TensorFlow) graph for a simple linear regression.
```shell
import tensorflow as tf

with tf.Graph() as single_device_graph:
  x_data = [1, 2, 3]
  y_data = [1, 2, 3]

  W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
  b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

  X = tf.placeholder(tf.float32, name="X")
  Y = tf.placeholder(tf.float32, name="Y")

  hypothesis = W * X + b

  cost = tf.reduce_mean(tf.square(hypothesis - Y))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
  train_op = optimizer.minimize(cost)
```

## `run` Function
Parallax allows flexible session call and logging as a normal TensorFlow application does. Users need to implement the `run` function as below.
```shell
def run(sess, num_iters, tensor_or_op_name_to_replica_names,
        num_workers, worker_id, num_replicas_per_worker)
```
* sess : `session` which can run the distributed graph created by Parallax.
* num_iters : The number of iterations to run. The number is received from the user but a different number can be used when comparing the throughput between PS and MPI.
* tensor_or_op_name_to_replica_names: A dictionary that maps the name of an operator or a tensor in the original graph to a list of names in the transformed graph. For the parameter server, Parallax does not change the shared operators like parameters and the parameter update operators while the operators that are needed for gradients computations are replicated as many times as the number of GPUs with the names including replica prefixes. MPI style transformation keeps all the names of operators so that the list length is always one.
* num_workers : The number of workers. It could be used for data partitioning or logging.
* worker_id : The worker id. It could be used for data partitioning or logging.
* num_replicas_per_worker: The number of replicas per worker. It could be used for data partitioning or logging.

Below code snippet comes from [TensorFlow CNN Benchmark](https://github.com/snuspl/parallax/blob/master/parallax/parallax/examples/tf_cnn_benchmarks/CNNBenchmark_distributed_driver.py) example.
```shell
def run(sess, num_iters, tensor_or_op_name_to_replica_names,
        num_workers, worker_id, num_replicas_per_worker):
	fetches = {
		'global_step':
			tensor_or_op_name_to_replica_names[bench.global_step.name][0],
		'cost': tensor_or_op_name_to_replica_names[bench.cost.name][0],
		'train_op':
			tensor_or_op_name_to_replica_names[bench.train_op.name][0],
	}
	fetched = sess.run(fetches)
```

## Data Partitioning
Parallax supports data parallelism, meaning that a disjoint subset of input data has to be assigned to each worker. The way of data processing is different according to the application. The way of data processing is different according to the application. The data processing could be defined as [dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) API, a python function, and operations in the graph. As a result, Parallax provides data partitioning for dataset API which is the simplest one, and utilizing other methods is possible as an additional option.

### 1. Utilizing the dataset API
TensorFlow introduces the dataset API for comfortable input preprocessing. The dataset is defined as a function library instead of operators in the TensorFlow graph, so Parallax uses a different approach compared to the examples above.
For the dataset, call `ds = shard.shard(ds)`. `ds` is a dataset defined by TensorFlow and `shard` is a Parallax library. Then, Parallax updates the function library definition related to the shard when the single graph is transformed. Note that Parallax does not support nested shard call, which means you cannot call `shard.shard(ds)` inside a map function of a dataset.
This is the input preprocessing code of [TensorFlow CNN Benchmark](https://github.com/snuspl/parallax/blob/master/parallax/parallax/examples/tf_cnn_benchmarks/preprocessing.py).
```shell
file_names.sort()
ds = tf.data.TFRecordDataset.list_files(file_names)
ds = shard.shard(ds)
ds = ds.apply(
    interleave_ops.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=10))
```

### 2. Using run function and feed
Some of the applications define input data as a placeholder, and feed them through TensorFlow session. In this case, you can utilize `num_workers`, `worker_id`, `num_replicas_per_worker` in the `run` function. This code snippet comes from [LM-1B](https://github.com/snuspl/parallax/blob/master/parallax/parallax/examples/lm1b/lm1b_distributed_driver.py) example.

```shell
def run(sess, num_iters, tensor_or_op_name_to_replica_names,
        num_workers, worker_id, num_replicas_per_worker):

        data_iterator = dataset.iterate_forever(FLAGS.batch_size * num_replicas_per_worker,
                                                FLAGS.num_steps, num_workers, worker_id)

        x_names = tensor_or_op_name_to_replica_names[model.x.name]
        y_names = tensor_or_op_name_to_replica_names[model.y.name]
        w_names = tensor_or_op_name_to_replica_names[model.w.name]
        for local_step in range(num_iters):
            x, y, w = next(data_iterator)
            feeds = {}
            for replica_id in range(num_replicas_per_worker):
                start_idx = FLAGS.batch_size * replica_id
                end_idx = FLAGS.batch_size * (replica_id + 1)
                feeds[x_names[replica_id]] = x[start_idx:end_idx]
                feeds[y_names[replica_id]] = y[start_idx:end_idx]
                feeds[w_names[replica_id]] = w[start_idx:end_idx]
            fetched = sess.run(fetches, feeds)
```
### 3. Embedding shard operators in the graph
Construct a single device graph with the `num_shards` and `shard_id` tensors from `shard.create_num_shards_and_shard_id()`. `shard` is a library from Parallax. Then, Parallax internally changes the tensor values while transforming the single graph into a distributed version. Below code comes from [Skip-Thoughts Vectors](https://github.com/snuspl/parallax/blob/master/parallax/parallax/examples/skip_thoughts/ops/input_ops.py).
```shell
data_files.sort()
num_files = len(data_files)
num_shards, shard_id = shard.create_num_shards_and_shard_id()
shard_size = num_files / num_shards
shard_size = tf.cast(shard_size, dtype=tf.int64)
remainder = num_files % num_shards

slice_begin = tf.cond(tf.less(shard_id, remainder + 1),
                      lambda: (shard_size + 1) * shard_id,
                      lambda: shard_size * shard_id + remainder)
slice_size = tf.cond(tf.less(shard_id, remainder), lambda: shard_size + 1,
                     lambda: shard_size)
data_files = tf.slice(data_files, [slice_begin], [slice_size])
```

## Setting Environment Variables
Parallax reads the environment variables below from the machine where an application is launched. `CUDA_VISIBLE_DEVICES` is set with the information from `resource_info` so that the user defined value will be ignored.
* PARALLAX_LOG_LEVEL : The log level of Parallax. Default level is INFO. Refer [logging](https://docs.python.org/2/library/logging.html).

We assume users set all other environment variables on each machine. However, SSH command which is used by Parallax for remote execution does not refer to environment variables defined in ~/.bashrc or ~/.profile. To set environment variables properly, you should set `~/.ssh/environment`(it requires `PermitUserEnvironment yes` in `/etc/ssh/sshd_config`, then call `/etc/init.d/ssh restart` to apply the modification).

## Running Parallax
Now, we are ready to run Parallax. Call Parallax `parallel_run` API as below. Check [here](parallax_api.md) for all the arguments.
```Shell
from parallax.core.python.common import runner
runner.parallel_run(single_device_graph, run_function, resource_info_file_path, number of iterations)
```
Then, execute the python file with `parallel_run`:
```Shell
$ python <python_file_name>
```
The command assumes that the source codebase is distributed and reachable in the same absolute path in each of the machines.

## Examples

More examples are here: [linear regression](/parallax/parallax/examples/simple), [TensorFlow CNN Benchmark](/parallax/parallax/examples/tf_cnn_benchmarks), [Skip-Thoughts Vectors](/parallax/parallax/examples/skip_thoughts), [LM-1B](/parallax/parallax/examples/lm1b), [NMT](/parallax/parallax/examples/nmt)


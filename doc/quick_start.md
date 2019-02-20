# Parallax Quick Start
Parallax mainly focuses on automatic parallelization of deep learning model graphs for a single device(CPU or GPU) with data parallelism. For automatic parallelization, Parallax requires some basic elements; resource information of the distributed environment, a single device graph, a data partitioning logic for data parallelism, a training method (asynchronous/synchronous), and a `run` function to invoke the computation of the distributed version of the graph.

## Resource Information
We assume a homogeneous environment where the number of GPUs per worker are equal. Parallax receives resource information as a file with the IP address and GPU ids(optional), as shown below. If the GPU ids are not served, Parallax detects all the GPUs in the host. Parallax runs `ps` tasks(for HYBRID and Parameter Server architectures) in all the hosts in the resource information file.
Note that AR architecture creates a worker for each GPU device, so the number of workers are the same as the number of GPUs even if workers are set with multiple GPUs. Parallax assumes that all hosts must be accessible by [ssh without password](http://www.linuxproblem.org/art_9.html)
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

single_device_graph = tf.Graph()
with single_device_graph.as_default_graph():
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

### 2. Feed Placeholder
Some of the applications define input data as a placeholder, and feed them through TensorFlow session. In this case, you can utilize `num_workers`, `worker_id`, `num_replicas_per_worker` from the `parallax.parallel_run` function. This code snippet comes from [LM-1B](https://github.com/snuspl/parallax/blob/master/parallax/parallax/examples/lm1b/lm1b_distributed_driver.py) example.
The value of feed dictionary must be a list as long as `num_replicas_per_worker`. Each element in the list is fed into a replica tensor in the distributed graph.

```shell
def run(sess, num_workers, worker_id, num_replicas_per_worker):

        data_iterator = dataset.iterate_forever(FLAGS.batch_size * num_replicas_per_worker,
                                                FLAGS.num_steps, num_workers, worker_id)

        for local_step in range(FLAGS.max_steps):
            x, y, w = next(data_iterator)
            feeds = {}
	    feeds[model.x] = np.split(x, num_replicas_per_worker)
	    feeds[model.y] = np.split(y, num_replicas_per_worker)
	    feeds[model.w] = np.split(w, num_replicas_per_worker)
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
## Variable partitioning
Parallax finds a near-optimal number of partitions to maximize parallelism while maintaining low computation and communication overhead. We support `tf.fixed_size_partitioner` with the number of partitions that Parallax finds using an internal cost model that predicts iteration time as a function of the number of partitions.
If some of the variables are assumed large enough to be partitioned, construct partitioner using `parallax.get_partitioner` with the minimum number of partitions possible without memory exceptions as an argument. By assigning `search_partitions` as True or False in ParallaxConfig, partitioning can be turned on or off. When there is a Parallax partitioner with `search_partitions=False`, the minimum number of partitions are used.
```shell
partitioner = parallax.get_partitioner(min_partitions)
with tf.variable_scope(
     "emb", partitioner=partitioner) as scope:
     emb_v = tf.get_variable(
          "emb_mat_var", [num_trainable_tokens, emb_size])
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


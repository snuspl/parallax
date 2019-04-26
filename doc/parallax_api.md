# Parallax API
This document explains Parallax API in detail. If you are a beginner of Parallax, follow [quick_start](quick_start.md).

## parallel_run

`parallel_run` transforms the `single_gpu_graph` for the distributed environment specified in the `resource_info` file with a specific communication method. This is either MPI (AR architecture), PS or HYBRID style communication. Then, it returns the `session` for running the transformed graph with `num_workers`, `worker_id` and `num_replicas_per_worker`
``` shell
def parallel_run(single_gpu_graph, resource_info,
                 sync=True, parallax_config=None)
```
* Args
  * single_gpu_graph: A complete TensorFlow graph that can run on a single device.
  * resource_info: A file which contains resource information.
  * sync: The training method(synchronous/asynchronous). `True` is the default.
  * [parallax_config](#parallaxconfig): The minor configuration for executing Parallax.

* Returns
  * session : Session, which is similiar to TensorFlow session, for running the graph of distributed version.
  * num_workers : The total number of workers.
  * worker_id : The worker id of the current process.
  * num_replicas_per_worker: The number of replicas in a worker. The value is always one for MPI and HYBRID.

### Graph Execution
The returned `session` from `parallel_run` is almost similiar with session in TensorFlow. You can feed and fetch values using operations/tensors or their names in the `single_gpu_graph`. However, the feed and fetch arguments are internally converted for distributed graph. For example, if you fetch `x` operation in the `single_gpu_graph`, the returned results will be a list of all the `x` operations in the worker. The length of the list is always the same as the `num_replicas_per_worker`. Feeding is similiar with fetching so that you must pass a list with values as many as `num_replicas_per_worker`.

Example (assume `num_replicas_per_worker` is 3)
```shell
    single_gpu_graph = tf.Graph()
    with single_gpu_graph.as_default():
      a = tf.placeholder(tf.float32, shape=[])
      x = tf.add(a, a)
      
    sess, _, _, num_replicas_per_worker = 
        parallax.parallel_run(single_gpu_graph,
	                      resource_info_file)
    result = sess.run(x, feed = {a: [1.0, 2.0, 3.0]})
    print(result) -> [2.0, 4.0, 6.0]
```

### ParallaxConfig
`parallax_config`, which is used in `parallel_run`, is an instance of **ParallaxConfig**, the minor configuration(**SessionConfig**, **CheckPointConfig**, **CommunicationConfig**) for executing Parallax.

```shell
ParallaxConfig(run_option=None, average_sparse=False, sess_config=None, redirect_path=None, 
               search_partitions=False, communication_config=CommunicationConfig(), 
               ckpt_config=CheckPointConfig())
```

* run_option:  A string(PS, MPI or HYBRID). The communication method for training.
* average_sparse: A boolean. If True, sparse parameters are updated by the averaged gradients over all replicas. Otherwise, the sum of all gradients are used.
* sess_config: The session configuration used in `tf.train.MonitoredTrainingSession`
* redirect_path: A string. Optional path to redirect logs as files.
* search_partitions: A boolean. If True and there is a Parallax partitioner, Parallax's automatic partitioning mechanism is activated. Otherwise, the partitioning is turned off.
* communication_config: The communication configuration for PS and MPI.
	* PSConfig
		* protocol: Specifies the protocol to be used by the server. Acceptable values include `"grpc"`, `"grpc+gdr"`, `"grpc+verbs"`, etc. `"grpc"` is the default.
		* replicate_variables: Each GPU has a copy of the variables, and updates its copy after the parameter servers are all updated with the gradients from all servers. Only works with `sync=True`.
                * local_aggregation: If True, gradients are aggregated within a machine before sending them to servers.
                * boundary_among_servers: Optimize operation placement among servers.
                * boundary_between_workers_and_servers: Optimize operation placement between workers and servers.
	* MPIConfig
		* use_allgatherv: Specifies whether to utilize OpenMPI `allgatherv` instead of NCCL `allgather`. `use_allgatherv=False` is recommended by Parallax.
		* mpirun_options: A string or list. Specifies the mpirun options, such as `-mca btl ^openib`. Empty string is the default.
* ckpt_config: The configuration for checkpoint.
	* ckpt_dir: The checkpoint directory to store/restore global variables.
	* save_ckpt_steps: The frequency, in number of global steps, that a checkpoint is saved using a default checkpoint saver.
	* save_ckpt_secs: The frequency, in seconds, that a checkpoint is saved using a default checkpoint saver.
* profile_config: The configuration for profile. CUPTI library path (e.g., /usr/local/cuda/extras/CUPTI/lib64) needs to be added to LD_LIBRARY_PATH.
	* profile_dir: The profile directory to store RunMetadata.
	* profile_steps: A list of steps when to store RunMetadata.
Below code is an example of how to use **ParallaxConfig**.
```
ckpt_config = parallax.CheckPointConfig(ckpt_dir=ckpt_dir,
                                        save_ckpt_steps=save_ckpt_steps)
ps_config = parallax.PSConfig(replicate_variables=replicate_variables,
                              protocol=protocol,
                              local_aggregation=local_aggregation,
                              boundary_among_servers=boundary_among_servers,
                              boundary_between_workers_and_servers=boundary_between_workers_and_servers)
mpi_config = parallax.MPIConfig(use_allgatherv=use_allgatherv,
                                mpirun_options=mpirun_options)
profile_config = parallax.ProfileConfig(profile_dir=/tmp/profile,
                                        profile_steps=[100,200,300])
parallax_config = parallax.Config()
parallax_config.run_option = run_option,
parallax_config.average_sparse = False
parallax_config.redirect_path = redirect_path
parallax_config.search_partitions = True
parallax_config.communication_config = parallax.CommunicationConfig(ps_config, mpi_config)
parallax_config.ckpt_config = ckpt_config
parallax_config.profile_config = profile_config
...
parallel_run(..., parallax_config=parallax_config)
```

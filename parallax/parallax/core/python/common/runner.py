# Copyright (C) 2018 Seoul National University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from multiprocessing.managers import BaseManager
import os
import Queue
import numpy as np
from scipy import optimize
import sys
import time
import traceback
import uuid
import signal

import tensorflow as tf
from tensorflow.core.protobuf import gradients_info_pb2
from tensorflow.core.framework import variable_pb2
import horovod.tensorflow as hvd

from parallax.core.python.common import graph_transform_lib
from parallax.core.python.common.lib import *
from parallax.core.python.common.config import ParallaxConfig
from parallax.core.python.common.consts import *
from parallax.core.python.mpi.runner import parallax_run_mpi
from parallax.core.python.mpi.runner import launch_mpi_driver
from parallax.core.python.ps.runner import parallax_run_ps
from parallax.core.python.ps.runner import launch_ps_driver
from parallax.core.python.hybrid.runner import parallax_run_hybrid
from parallax.core.python.hybrid.runner import launch_hybrid_driver

def _get_grads(single_gpu_meta_graph_def):
    trainable_vars = []
    trainable_vars_defs = single_gpu_meta_graph_def.collection_def[tf.GraphKeys.TRAINABLE_VARIABLES]
    for var_def_string in trainable_vars_defs.bytes_list.value:
        var_def = variable_pb2.VariableDef()
        var_def.ParseFromString(var_def_string)
        trainable_vars.append(var_def.variable_name)
    sparse_grads = []
    dense_grads = []
    grad_info_defs = single_gpu_meta_graph_def.collection_def[tf.GraphKeys.GRADIENTS_INFO]
    for grad_info_def_string in grad_info_defs.bytes_list.value:
        gradients_info_def = gradients_info_pb2.GradientsInfoDef()
        gradients_info_def.ParseFromString(grad_info_def_string)
        if gradients_info_def.target_tensor_info.values_tensor_name not in trainable_vars:
            continue
        if gradients_info_def.grad_tensor_info.tensor_type == gradients_info_pb2.GradientsInfoDef.TensorInfoDef.INDEXED_SLICES:
            sparse_grads.append(gradients_info_def)
        else:
            dense_grads.append(gradients_info_def)
    assert len(sparse_grads) > 0 or len(dense_grads) > 0
    return sparse_grads, dense_grads

class PartitionStatCollector(object):

    def __init__(self, p_to_test, address):
        self.p_to_test = p_to_test
        self.address = address
        self.prev_p = None
        self.prev_exec_time = None
        self.exec_time_list = []
        self.p_list = []
        self.start = None
        self.min_partitions = p_to_test
        
    def setup_manager(self):
        if self.start is None:
            self.start = time.time()
        self.m = BaseManager(address=self.address, authkey='parallax_auth')
        queue = Queue.Queue()
        BaseManager.register('queue', callable=lambda:queue)
        self.m.start()
        return self.m

    def recv_exec_time(self, processes, cleanup, num_required):
        stop = False
        worker_exec_times = []
        all_alive = True
        while len(worker_exec_times) != num_required and all_alive:
            time.sleep(10)
            q = self.m.queue()
            while q.qsize() > 0:
                worker_exec_times.append(q.get())
 
            for p in processes:
              if p.poll() is not None:
                  all_alive = False
                  break

        cleanup(None, None)
        time.sleep(10)

        if all_alive:
            curr_p = self.p_to_test
            curr_exec_time = np.mean(worker_exec_times)
            self.p_list.append(curr_p)
            self.exec_time_list.append(curr_exec_time)

            if self.prev_p:
		if self.prev_exec_time < curr_exec_time:
		    # decrease or stop
                    if self.prev_p > curr_p:
                        stop = True
		    else:
	                # search the oposite partitions
			self.p_to_test = min(self.p_list) / 2
		else:
		    # keep increase or keep decrease
		    if self.prev_p < curr_p:
                        if (self.prev_exec_time / curr_exec_time) < 0.1:
                            self.p_to_test = min(self.p_list) / 2
                        else:
			    self.p_to_test *= 2
		    else:
			self.p_to_test /= 2
                        if (self.prev_exec_time / curr_exec_time) < 0.1:
                            stop = True

		if self.p_to_test < self.min_partitions:
		    stop = True
	    else:
		# increase first
		self.p_to_test *= 2

	    self.prev_p = curr_p
	    self.prev_exec_time = curr_exec_time
        else:
            # communication error when num partitions is small
            if self.prev_p:
                stop = True
            else:
                self.p_to_test *= 2
                self.min_partitions = self.p_to_test

        if stop:
            end = time.time()
            self.p_to_test = self._find_optimal_p()
            parallax_log.info('optimal partitions: %d, search time: %d secs' % \
                (self.p_to_test, end - self.start))
            print('optimal partitions: %d, search time: %d secs' % \
                (self.p_to_test, end - self.start))

        return not stop, self.p_to_test

    def _find_optimal_p(self):
        parallax_log.info('start finding optimal p')
        print('start finding optimal p')
        parallax_log.info(self.p_list)
        print(self.p_list)
        parallax_log.info(self.exec_time_list)
        print(self.exec_time_list)
        
        if len(self.p_list) < 3:
          min_exec_time = min(self.exec_time_list)
          return self.p_list[self.exec_time_list.index(min_exec_time)]
            
        max_time = float(max(self.exec_time_list))
        exec_times = [t / max_time for t in self.exec_time_list]

        fitfunc = lambda n, a, b, c: b / n + a * (n - 1) + c
        p, pcov = optimize.curve_fit(fitfunc, np.array(self.p_list), np.array(exec_times))

        min_p = min(self.p_list)
        max_p = max(self.p_list)

        min_exec_time = None
        optimal_p = None
        for i in range(min_p, max_p + 1):
          prediction = fitfunc(i, p[0], p[1], p[2])

          if min_exec_time is None or min_exec_time > prediction:
            min_exec_time = prediction
            optimal_p = i

        return optimal_p

def _parallax_run_master(single_gpu_meta_graph_def,
                         config):

    # Get caller's file path, have to find a better way for this.
    driver_path = os.path.abspath(sys.argv[0])

    # Get user-defined command line args
    args = sys.argv[1:]

    sparse_grads, dense_grads = _get_grads(single_gpu_meta_graph_def)

    search_p = False
    p_to_test = None
    if config.search_partitions:
      # Set to find automatic embedding partitoning
      p_to_test = len(config.resource_info['worker'])
      address = (config.resource_info['master'][0]['hostname'],
                 int(config.resource_info['master'][0]['port'][0]))
      
      stat_collector = PartitionStatCollector(p_to_test, address)
      search_p = True

    cleanup = None
    try:
        while True:
            m = None
            if search_p:
                m = stat_collector.setup_manager()

	    if config.run_option == 'MPI' or \
		(config.run_option == 'HYBRID' and len(sparse_grads) == 0):
                num_workers = sum([max(1, len(w['gpus'])) for w in config.resource_info['worker']])
		processes, cleanup = \
			launch_mpi_driver(driver_path,
					  args,
					  config,
                                          p_to_test,
                                          m)
	    elif config.run_option == 'PS' or \
		(config.run_option == 'HYBRID' and len(dense_grads) == 0):
                num_workers = len(config.resource_info['worker'])
		processes, logfiles, cleanup = \
			launch_ps_driver(driver_path,
					 args,
					 config,
                                         p_to_test,
                                         m)
	    elif config.run_option == 'HYBRID':
                num_workers = sum([max(1, len(w['gpus'])) for w in config.resource_info['worker']])
		processes, cleanup = \
		    launch_hybrid_driver(driver_path,
					 args,
					 config,
                                         p_to_test,
                                         m)
                
            if not search_p:
                processes[0].wait()
                break
            else:
                search_p, p_to_test = \
                    stat_collector.recv_exec_time(processes, cleanup, num_workers)
    except:
        traceback.print_exc()
    finally:
        if cleanup is not None:
            try:
                cleanup(None, None)
            except:
                parallax_log.debug("master runner ends")

def parallel_run(single_gpu_graph,
                 resource_info,
                 sync=True,
                 parallax_config=ParallaxConfig()):
    """Invokes the `run` function to run the `single gpu graph` on
       distributed environment specified in the `resource_info` file
       with a specific communication method. This is either MPI or the
       PS style communication, whichever is faster in synchronous
       training, and PS style communication for asynchronous training.

    Args:
      single_gpu_graph: A complete TensorFlow graph that can run on a
        single device.
      resource_info: Path to the file that contains the resource information.
      sync: The training method(synchronous/asynchronous).
        `True` is the default.
      parallax_config: `ParallaxConfig` object for tunning the behavior of
        Parallax.
    """

    run_option = parallax_config.run_option
    if run_option not in ['PS', 'MPI', 'HYBRID']:
      raise ValueError('run_option must be PS, MPI or HYBRID')

    if not sync and (run_option == 'MPI' or run_option == 'HYBRID'):
        raise ValueError('sync must be True if run_option is MPI or HYBRID')    

    parallax_run_option = os.getenv(PARALLAX_RUN_OPTION, PARALLAX_RUN_MASTER)
    single_gpu_meta_graph_def = \
        tf.train.export_meta_graph(graph=single_gpu_graph)
    parallax_log.info('parallel_run(%s)', parallax_run_option)

    if parallax_run_option == PARALLAX_RUN_MASTER:
      resource_info = parse_resource_info(resource_info, run_option)
    else:
      parallax_log.info('resource %s', os.getenv(PARALLAX_RESOURCE_INFO))
      resource_info = deserialize_resource_info(os.getenv(PARALLAX_RESOURCE_INFO))

    parallax_config.set_sync(sync)
    parallax_config.set_resource_info(resource_info)

    kwargs = {
        'single_gpu_meta_graph_def': single_gpu_meta_graph_def,
        'config': parallax_config,
    }

    if parallax_run_option == PARALLAX_RUN_MASTER:
         _parallax_run_master(**kwargs)
         sys.exit()
    elif parallax_run_option == PARALLAX_RUN_MPI:
        return parallax_run_mpi(**kwargs)
    elif parallax_run_option == PARALLAX_RUN_PS:
        return parallax_run_ps(**kwargs)
    elif parallax_run_option == PARALLAX_RUN_HYBRID:
        return parallax_run_hybrid(**kwargs)

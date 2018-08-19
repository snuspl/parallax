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

import os
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

def _parallax_run_master(single_gpu_meta_graph_def,
                         run,
                         config):

    # Get caller's file path, have to find a better way for this.
    driver_path = os.path.abspath(sys.argv[0])

    # Get user-defined command line args
    args = sys.argv[1:]

    sparse_grads, dense_grads = _get_grads(single_gpu_meta_graph_def)
    cleanup = None
    try:
        if config.run_option == 'MPI' or \
            (config.run_option == 'HYBRID' and len(sparse_grads) == 0):

            process, cleanup = \
                    launch_mpi_driver(driver_path,
                                      args,
                                      config)
            process.wait()
        elif config.run_option == 'PS' or \
            (config.run_option == 'HYBRID' and len(dense_grads) == 0):
            chief_worker_process, logfiles, cleanup = \
                    launch_ps_driver(driver_path,
                                     args,
                                     config)
            chief_worker_process.wait()
        elif config.run_option == 'HYBRID':
            process, cleanup = \
                launch_hybrid_driver(driver_path,
                                     args,
                                     config)
            process.wait()
    except:
        traceback.print_exc()
    finally:
        if cleanup is not None:
            try:
                cleanup(None, None)
            except:
                parallax_log.debug("master runner ends")


def parallel_run(single_gpu_graph,
                 run,
                 resource_info,
                 num_iterations,
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
      run: A function which runs the transformed graph in a distributed
        environment.
      resource_info: Path to the file that contains the resource information.
      num_iterations: The number of iterations to be run on each worker.
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
    parallax_config.set_num_iterations(num_iterations)
    parallax_config.set_resource_info(resource_info)

    kwargs = {
        'single_gpu_meta_graph_def': single_gpu_meta_graph_def,
        'run': run,
        'config': parallax_config,
    }

    if parallax_run_option == PARALLAX_RUN_MASTER:
        _parallax_run_master(**kwargs)
    elif parallax_run_option == PARALLAX_RUN_MPI:
        parallax_run_mpi(**kwargs)
    elif parallax_run_option == PARALLAX_RUN_PS:
        parallax_run_ps(**kwargs)
    elif parallax_run_option == PARALLAX_RUN_HYBRID:
        parallax_run_hybrid(**kwargs)

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
import horovod.tensorflow as hvd

from parallax.core.python.common import graph_transform_lib
from parallax.core.python.common.lib import *
from parallax.core.python.common.config import ParallaxConfig
from parallax.core.python.common.consts import *
from parallax.core.python.mpi.runner import parallax_run_mpi
from parallax.core.python.mpi.runner import launch_mpi_driver
from parallax.core.python.ps.runner import parallax_run_ps
from parallax.core.python.ps.runner import launch_ps_driver


def _parallax_run_master(single_gpu_meta_graph_def,
                         run,
                         config):

    if not config.sync and config.run_option == 'MPI':
      raise ValueError("MPI is only possible with synchronous training.")

    # Get caller's file path, have to find a better way for this.
    driver_path = os.path.abspath(sys.argv[0])

    # Get user-defined command line args
    args = sys.argv[1:]

    cleanup = None
    try:
        if config.sync and config.run_option is None:
            # Test MPI
            process, cleanup = \
                launch_mpi_driver(driver_path,
                                  args,
                                  config,
                                  is_test=True)
            num_workers = 0
            for worker in config.resource_info['worker']:
                if len(worker['gpus']) > 0:
                    num_workers += len(worker['gpus'])
                else:
                    num_workers += 1
            mpi_exec_time = \
                get_average_execution_time(config.resource_info['master'][0],
                                           num_workers)

            # kill processes if the chief worker receives average
            # exectution time using MPI
            os.killpg(os.getpgid(process.pid) ,signal.SIGINT)

            # Test PS
            chief_worker_process, logfiles, cleanup = \
                launch_ps_driver(driver_path,
                                 args,
                                 config,
                                 is_test=True)
            num_workers = len(config.resource_info['worker'])
            ps_exec_time = \
                get_average_execution_time(config.resource_info['master'][0],
                                           num_workers)

            # kill processes if the chief worker receives average
            # exectution time using PS
            cleanup(None, None)

            parallax_log.debug('mpi exec time : %d secs, \
                               ps exec time: %d secs'
                               % (mpi_exec_time, ps_exec_time))

            time.sleep(10)

            # Select MPI
            if mpi_exec_time < ps_exec_time:
                process, cleanup = \
                    launch_mpi_driver(driver_path,
                                      args,
                                      config,
                                      is_test=False)
                process.wait()
            # Select PS
            else:
                chief_worker_process, logfiles, cleanup = \
                    launch_ps_driver(driver_path,
                                     args,
                                     config,
                                     is_test=False)
                chief_worker_process.wait()
        elif config.run_option == 'MPI':
            process, cleanup = \
                    launch_mpi_driver(driver_path,
                                      args,
                                      config,
                                      is_test=False)
            process.wait()
        elif config.run_option == 'PS':
            chief_worker_process, logfiles, cleanup = \
                    launch_ps_driver(driver_path,
                                     args,
                                     config,
                                     is_test=False)
            chief_worker_process.wait()
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

    parallax_run_option = os.getenv(PARALLAX_RUN_OPTION, PARALLAX_RUN_MASTER)
    single_gpu_meta_graph_def = \
        tf.train.export_meta_graph(graph=single_gpu_graph)
    parallax_log.info('parallel_run(%s)', parallax_run_option)

    if parallax_run_option == PARALLAX_RUN_MASTER:
      resource_info = parse_resource_info(resource_info)
    else:
      parallax_log.info('resource %s', os.getenv(PARALLAX_RESOURCE_INFO))
      resource_info = deserialize_resource_info(os.getenv(PARALLAX_RESOURCE_INFO))

    if parallax_run_option == PARALLAX_TEST_MPI or parallax_run_option == PARALLAX_TEST_PS:
        num_iterations = NUM_ITERATIONS_FOR_TEST

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
    elif parallax_run_option == PARALLAX_TEST_MPI:
        parallax_run_mpi(is_test=True, **kwargs)
    elif parallax_run_option == PARALLAX_RUN_MPI:
        parallax_run_mpi(is_test=False, **kwargs)
    elif parallax_run_option == PARALLAX_TEST_PS:
        parallax_run_ps(is_test=True, **kwargs)
    elif parallax_run_option == PARALLAX_RUN_PS:
        parallax_run_ps(is_test=False, **kwargs)

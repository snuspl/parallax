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

from __future__ import print_function

import signal
import subprocess
import time
import logging
import getpass
import os
import uuid
from functools import reduce

import tensorflow as tf
import horovod.tensorflow as hvd

from parallax.core.python.common.lib import *
from parallax.core.python.common.consts import *
from parallax.core.python.common.partitions import *
from parallax.core.python.common.session_context import ParallaxSessionContext
from parallax.core.python.mpi.graph_transform import graph_transform_mpi

def create_mpi_script(driver_path, args, hostname, gpus, partitions, search,
                      port=22):

    cmd = 'ssh -p %d %s "mkdir -p %s"' % (port, hostname, REMOTE_PARALLAX_ROOT)
    parallax_log.warning(colored('\n$ %s' % cmd, 'red'))
    proc = subprocess.Popen(args=cmd, shell=True)
    proc.wait()

    cmd_run = 'python %s %s' % (driver_path, ' '.join(args))

    try:
        parallax_log_level = os.environ['PARALLAX_LOG_LEVEL']
    except:
        parallax_log_level = logging.INFO
    env = {
        "CUDA_VISIBLE_DEVICES": ','.join(str(gpuid) for gpuid in gpus),
        "PARALLAX_LOG_LEVEL": parallax_log_level,
        PARALLAX_HOSTNAME: hostname,
        PARALLAX_SEARCH: search,
    }
    if partitions:
         env[PARALLAX_PARTITIONS] = partitions

    cmd_env = ' '.join(
        map(lambda k: 'export %s=%s;' % (k[0], k[1]), env.items()))
    try:
        cmd_venv = ' source %s/bin/activate; '\
                    % os.environ['VIRTUAL_ENV']
        full_cmd = ' '.join([cmd_env, cmd_venv, cmd_run])
    except:
        full_cmd = ' '.join([cmd_env, cmd_run])
    mpi_script = 'bash -c \"%s\"' % full_cmd

    remote_cmd = 'echo \'%s\' | ' % mpi_script
    remote_cmd += 'ssh -p %d %s' % (port, hostname)
    remote_cmd += ' \'cat > %s; chmod 777 %s\'' % (REMOTE_MPI_SCRIPT_PATH, REMOTE_MPI_SCRIPT_PATH)
    parallax_log.warning(colored('\n$ %s' % remote_cmd, 'red'))
    proc = subprocess.Popen(args=remote_cmd, shell=True)
    proc.wait()


def _prepare_workers(workers, driver_path, args, partitions, search):
    for worker in workers:
        _prepare_worker(worker, driver_path, args, partitions, search)


def _prepare_worker(worker, driver_path, args, partitions, search):
    create_mpi_script(driver_path, args, worker['hostname'], worker['gpus'],
                      partitions, search)


def _get_mpi_cmd(config):
    workers = config.resource_info['worker']
    mpi_cmd = 'mpirun -bind-to none -map-by slot' \
              ' -mca orte_base_help_aggregate 0'\
              ' -x NCCL_DEBUG=INFO '
    mpi_cmd += config.communication_config.mpi_config.mpirun_options

    arg_runop = '-x %s=%s' % (PARALLAX_RUN_OPTION,
                              PARALLAX_RUN_MPI)
    arg_resource = '-x %s=%s' % (PARALLAX_RESOURCE_INFO, serialize_resource_info(config.resource_info))
    num_process = reduce(lambda s, x: s + len(x['gpus']), workers, 0)
    arg_np = '-np %d' % num_process
    arg_host = '-H %s' % get_cluster_str_for_hosts(workers, with_slots=True)
    arg_out = ''
    if config.redirect_path is not None:
        arg_out += '-output-filename %s/mpi' % config.redirect_path
    arg_script = 'bash %s' % REMOTE_MPI_SCRIPT_PATH
    std_err_redir = '2>&1'

    mpi_cmd = ' '.join([mpi_cmd, arg_runop, arg_resource, arg_np, arg_host,
                        arg_out, arg_script, std_err_redir])

    return mpi_cmd


def launch_mpi_driver(driver_path, args, config, partitions, m):
    workers = config.resource_info['worker']
    _prepare_workers(workers, driver_path, args, partitions, m is not None)

    mpi_cmd = _get_mpi_cmd(config)

    parallax_log.warning(colored('\n$ %s' % mpi_cmd, 'red'))
    proc = subprocess.Popen(args=mpi_cmd, shell=True, preexec_fn=os.setsid)

    def cleanup_mpi(recv_signal, frame):
        if m:
            m.shutdown()
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except:
            pass

    signal.signal(signal.SIGINT, cleanup_mpi)
    return [proc], cleanup_mpi


def _init_global_vars(sess):
    hvd_bcast_global_vars_op = tf.get_default_graph() \
        .get_operation_by_name('auto_parallel_bcast_global_vars')
    if hvd_bcast_global_vars_op is not None:
        for c in hvd_bcast_global_vars_op.control_inputs:
            sess.run(c)


def parallax_run_mpi(single_gpu_meta_graph_def, config):
    hostname = os.getenv(PARALLAX_HOSTNAME, 0)
    create_profile_directory(config.profile_config.profile_dir,
                             config.resource_info, hostname)

    mpi_meta_graph_def, tensor_or_op_name_to_replica_names = \
        graph_transform_mpi(single_gpu_meta_graph_def, config)
    worker_id = hvd.rank()
    num_workers = hvd.size()
        
    if config.profile_config.profile_dir:
        append_task_info(config.profile_config.profile_dir,
                         hostname,
                         ['worker:%d'%worker_id])

    with tf.Graph().as_default() as graph_to_run:
        parallax_log.debug("Importing MPI graph on worker %d" % worker_id)
        tf.train.import_meta_graph(mpi_meta_graph_def)

        if config.export_graph_path:
            export_meta_graph(config.export_graph_path, worker_id)

        if config.profile_config.profile_dir:
            path = os.path.join(config.profile_config.profile_dir, hostname,
                                'worker:%d'%worker_id)
            export_meta_graph(path, worker_id)
            
            if config.profile_config.profile_worker != None and worker_id != config.profile_config.profile_worker:
                #Only one CUPTI profiler can run in a machine
                #See tensorflow/tensorflow/core/platform/default/device_tracer.cc:L452
                config.profile_config.profile_dir = None
            else:
                config.profile_config.profile_dir = \
                    os.path.join(config.profile_config.profile_dir, hostname,
                                 'worker:%d'%worker_id, 'run_meta')      
 
        ckpt_hooks = build_ckpt_hooks(config.get_ckpt_config()) if worker_id == 0 else None

        sess_config = config.sess_config
        if sess_config is None:
            sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
        sess = tf.train.MonitoredTrainingSession(
                is_chief=True,
                checkpoint_dir=config.get_ckpt_config().ckpt_dir if worker_id == 0 else None,
                # TODO: Allow user-defined hooks
                hooks=None,
                chief_only_hooks=ckpt_hooks,
                save_checkpoint_secs=None,
                save_summaries_steps=None,
                save_summaries_secs=None,
                config=sess_config)

        parallax_log.debug(
            "Created MonitoredTrainingSession for worker %d" % worker_id)
        _init_global_vars(sess)
        parallax_log.debug(
            "Finished initialization process, start training on \
             worker %d" % worker_id)
        step = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0])
        sess_context = \
            ParallaxSessionContext(step,
                                   tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0],
                                   config.profile_config.profile_dir,
                                   config.profile_config.profile_steps,
                                   config.profile_config.profile_range,
                                   tensor_or_op_name_to_replica_names,
                                   1,
                                   config.resource_info['master'][0])
        sess_context.set_parallax_session_context()
        return sess, num_workers, worker_id, 1

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
import os
from functools import reduce

import tensorflow as tf
import horovod.tensorflow as hvd

from parallax.core.python.common.lib import *
from parallax.core.python.common.consts import *
from parallax.core.python.common.session_context import ParallaxSessionContext
from parallax.core.python.hybrid.graph_transform import graph_transform_hybrid
from parallax.core.python.ps.runner import launch_ps

def create_mpi_script(driver_path, args, hostname, gpus, resource_info, machine_id, port=22):
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
        PARALLAX_MACHINE_ID: machine_id,
        PARALLAX_HOSTNAME: hostname,
        "PARALLAX_RESOURCE_INFO": resource_info,
    }

    cmd_env = ' '.join(
        map(lambda (k, v): 'export %s=%s;' % (k, v), env.iteritems()))
    try:
        cmd_venv = ' source %s/bin/activate; '\
                    % os.environ['VIRTUAL_ENV_PATH']
        full_cmd = ' '.join([cmd_env, cmd_venv, cmd_run])
    except:
        full_cmd = ' '.join([cmd_env, cmd_run])
    mpi_script = 'bash -c \"%s\"' % full_cmd

    remote_cmd = 'echo \'%s\' | ' % mpi_script
    remote_cmd += 'ssh -p %d %s' % (port, hostname)
    remote_mpi_script_path = os.path.join(REMOTE_PARALLAX_ROOT, 'mpi_run_%d.sh' % machine_id)
    remote_cmd += ' \'cat > %s\' && chmod 777 %s' % (remote_mpi_script_path, remote_mpi_script_path)
    print(colored('\n$ %s' % remote_cmd, 'red'))
    proc = subprocess.Popen(args=remote_cmd, shell=True)
    proc.wait()


def _prepare_workers(workers, driver_path, args, resource_info):
    for i, worker in enumerate(workers):
        _prepare_worker(worker, driver_path, args, resource_info, i)


def _prepare_worker(worker, driver_path, args, resource_info, machine_id):
    create_mpi_script(driver_path, args, worker['hostname'], worker['gpus'],
                      resource_info, machine_id)


def _get_hybrid_cmd(workers, protocol, redirect_path, mpi_cmd_in_config):
    mpi_cmd = 'mpirun -bind-to none -map-by slot' \
              ' -mca orte_base_help_aggregate 0'\
              ' -x NCCL_DEBUG=INFO'
    arg_runop = '-x %s=%s' % (PARALLAX_RUN_OPTION, PARALLAX_RUN_HYBRID)
    num_process = reduce(lambda s, x: s + max(len(x['gpus']), 1), workers, 0)
    arg_np = '-np %d' % num_process
    arg_host = '-H %s' % get_cluster_str_for_hosts(workers, with_slots=True)
    arg_redir = '-output-filename %s' % os.path.join(redirect_path, 'worker') \
        if redirect_path is not None else ''
    arg_script = 'bash %s' % REMOTE_MPI_SCRIPT_PATH
    std_err_redir = '2>&1'

    mpi_cmd = ' '.join([mpi_cmd, mpi_cmd_in_config, arg_runop, arg_np, arg_host,
                        arg_redir, arg_script, std_err_redir])

    return mpi_cmd

def launch_hybrid_driver(driver_path, args, config):
    resource_info = config.resource_info
    resource_info_file = serialize_resource_info(config.resource_info)
    protocol = config.communication_config.ps_config.protocol
    redirect_path = config.redirect_path   

    workers = config.resource_info['worker']
    _prepare_workers(workers, driver_path, args, resource_info_file)

    mpi_command = config.communication_config.mpi_config.mpirun_options
    hybrid_cmd = _get_hybrid_cmd(workers, protocol, redirect_path, mpi_command)

    processes = []
    print(colored('\n$ %s' % hybrid_cmd, 'red'))
    proc = subprocess.Popen(args=hybrid_cmd, shell=True)

    pss = resource_info['ps'] if 'ps' in resource_info else []
    for ps_id in range(len(pss)):
        ps_proc, ps_logs = \
            launch_ps(ps_id, config)
        processes.append(ps_proc)

    def cleanup(recv_signal, frame):
        for process in processes:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)

    signal.signal(signal.SIGINT, cleanup)
    return proc, cleanup


def _init_global_vars(sess):
    hvd_bcast_global_vars_op = tf.get_default_graph() \
        .get_operation_by_name('auto_parallel_bcast_global_vars')
    if hvd_bcast_global_vars_op is not None:
        control_inputs = [c.name for c in hvd_bcast_global_vars_op.control_inputs]
        control_inputs.sort()
        for c in control_inputs:
            sess.run(c)

def _get_worker_info():
    machine_id = int(os.getenv(PARALLAX_MACHINE_ID, -1))
    if machine_id == -1:
        raise RuntimeError(
            "Need to set environment variable PARALLAX_MACHINE_ID")
    hostname = os.getenv(PARALLAX_HOSTNAME, 0)
    if hostname is None:
        raise RuntimeError(
            "Need to set environment variable PARALLAX_HOSTNAME")
    return machine_id, hostname

def get_tf_clusterspec_for_hybrid(resource_info):
    tf_cluster_dict = {}
    for job in ['ps', 'worker']:
        if job not in resource_info:
            continue
        hosts = resource_info[job]
        tf_cluster_dict[job] = []
        for host in hosts:
            for i in range(max(len(host['gpus']), 1)):
                tf_cluster_dict[job].append(
                    '%s:%d' % (host['hostname'], host['port'][i]))
    cluster_spec = tf.train.ClusterSpec(tf_cluster_dict)
    return cluster_spec

def parallax_run_hybrid(single_gpu_meta_graph_def,
                        config):

    # Initialize horovod
    hvd.init()
    #worker_id = hvd.rank()
    local_worker_id = hvd.local_rank()
    num_workers = hvd.size()

    machine_id, hostname = _get_worker_info()

    sess_config = config.sess_config
    if sess_config is None:
        sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
    cluster_spec = get_tf_clusterspec_for_hybrid(config.resource_info)
    worker_id = 0
    for i in range(machine_id):
      worker_id += max(1, len(config.resource_info['worker'][i]['gpus']))
    worker_id += hvd.local_rank()
    server = tf.train.Server(cluster_spec, job_name='worker',
                             task_index=worker_id, protocol=config.communication_config.ps_config.protocol,
                             config=sess_config)
    
    meta_graph_def, tensor_or_op_name_to_replica_names = graph_transform_hybrid(
        single_gpu_meta_graph_def,
        worker_id,
        local_worker_id,
        machine_id,
        hostname,
        config)

    with tf.Graph().as_default() as graph_to_run:
        parallax_log.debug("Importing MPI graph on worker %d" % worker_id)

        tf.train.import_meta_graph(meta_graph_def)
        if config.export_graph_path:
            export_hybrid_meta_graph(config.export_graph_path, worker_id)

        ckpt_hooks = \
            build_ckpt_hooks(config.get_ckpt_config()) \
            if worker_id == 0 else None

        sess = tf.train.MonitoredTrainingSession(
                master=server.target,
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
            "Finished initialization process, start training on worker %d"
            % worker_id)

        step = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0])
        sess_context = \
            ParallaxSessionContext(step,
                                   config.profile_config.profile_dir,
                                   config.profile_config.profile_steps,
                                   tensor_or_op_name_to_replica_names,
                                   1)
        sess_context.set_parallax_session_context()
        return sess, num_workers, worker_id, 1

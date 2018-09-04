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
import time
import logging
import getpass
import os
import uuid

import tensorflow as tf

from parallax.core.python.common import graph_transform_lib
from parallax.core.python.common.lib import *
from parallax.core.python.common.consts import *
from parallax.core.python.common.session_context import ParallaxSessionContext
from parallax.core.python.ps.graph_transform import graph_transform_ps


def _create_log_files(redirect_path, job, task_id):
    directory = os.path.dirname(redirect_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = '%s/ps' % redirect_path
    if not os.path.exists(directory):
        os.makedirs(directory)

    stdout = open(os.path.join(directory, 'log_%s%d_stdout' % (job, task_id)),
                  'w')
    stderr = open(os.path.join(directory, 'log_%s%d_stderr' % (job, task_id)),
                  'w')
    return stdout, stderr


def _prepare_ps(ps_info):
    remote_copy(ps_info['hostname'], LOCAL_LAUNCH_PS_PATH,
                REMOTE_PARALLAX_ROOT)


def _get_launch_ps_cmd(task, config):
    cmd = 'python %s --job_name=%s --task_index=%d --protocol=%s' % (
        REMOTE_LAUNCH_PS_PATH,
        'ps',
        task,
        config.communication_config.ps_config.protocol)

    cmd += ' --ps_hosts=%s' % get_cluster_str_for_hosts(config.resource_info['ps'],
                                                            with_slots=False)
    cmd += ' --worker_hosts=%s' % get_cluster_str_for_hosts(
        config.resource_info['worker'], with_slots=False)

    return cmd


def _get_ps_env(ps_info, config):
    try:
        parallax_log_level = os.environ['PARALLAX_LOG_LEVEL']
    except:
        parallax_log_level = logging.INFO
    env = {
        "CUDA_VISIBLE_DEVICES": ','.join(
            str(gpuid) for gpuid in ps_info['gpus']),
        "PARALLAX_LOG_LEVEL": parallax_log_level,
        "PARALLAX_RESOURCE_INFO": serialize_resource_info(config.resource_info),
    }

    return env


def launch_ps(task, config):
    ps_info = config.resource_info['ps'][task]
    _prepare_ps(ps_info)

    cmd = _get_launch_ps_cmd(task, config)
    env = _get_ps_env(ps_info, config)

    # TODO: better mechanism for managing log files
    if config.redirect_path is not None:
        stdout, stderr = _create_log_files(config.redirect_path, 'ps', task)
        logfiles = [stdout, stderr]
    else:
        stdout, stderr = None, None
        logfiles = []
    try:
        python_venv = os.environ['VIRTUAL_ENV']
    except:
        python_venv = None
    return remote_exec(cmd, ps_info['hostname'], stdout, stderr, env,
                       python_venv=python_venv),\
        logfiles


def _prepare_worker(worker_info):
    pass


def _get_launch_worker_cmd(driver_path, args):
    cmd = 'python %s %s' % (driver_path, ' '.join(args))
    return cmd


def _get_worker_env(worker_id, config):
    workers = config.resource_info['worker']
    worker_info = workers[worker_id]
    num_workers = len(workers)
    try:
        parallax_log_level = os.environ['PARALLAX_LOG_LEVEL']
    except:
        parallax_log_level = logging.INFO
    env = {
        "CUDA_VISIBLE_DEVICES": ','.join(
            str(gpuid) for gpuid in worker_info['gpus']),
        "PARALLAX_LOG_LEVEL": parallax_log_level,
        PARALLAX_RUN_OPTION: PARALLAX_RUN_PS,
        PARALLAX_RESOURCE_INFO: serialize_resource_info(config.resource_info),
        PARALLAX_WORKER_ID: worker_id,
        PARALLAX_NUM_WORKERS: num_workers
    }

    return env


def launch_worker(driver_path, args, worker_id, config):
    worker_info = config.resource_info['worker'][worker_id]

    cmd = _get_launch_worker_cmd(driver_path, args)
    env = _get_worker_env(worker_id, config)

    # TODO: better mechanism for managing log files
    if config.redirect_path is not None:
        stdout, stderr = _create_log_files(config.redirect_path, 'worker', worker_id)
        logfiles = [stdout, stderr]
    else:
        stdout, stderr = None, None
        logfiles = []

    try:
        python_venv = os.environ['VIRTUAL_ENV']
    except:
        python_venv = None
    return remote_exec(cmd, worker_info['hostname'], stdout, stderr, env=env,
                       python_venv=python_venv), \
        logfiles


def launch_ps_driver(driver_path, args, config):
    workers = config.resource_info['worker']
    pss = config.resource_info['ps'] if 'ps' in config.resource_info else []

    logfiles = []
    processes = []
    chief_worker_process = None
    for worker_id in range(len(workers)):
        worker_proc, worker_logs =\
            launch_worker(driver_path, args, len(workers) - worker_id - 1,
                          config)
        logfiles += worker_logs
        if worker_id == 0:
            chief_worker_process = worker_proc
        processes.append(worker_proc)

    for ps_id in range(len(pss)):
        ps_proc, ps_logs = launch_ps(ps_id, config)
        logfiles += ps_logs
        processes.append(ps_proc)

    def cleanup_ps(recv_signal, frame):
        for process in processes:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)

    signal.signal(signal.SIGINT, cleanup_ps)
    return chief_worker_process, logfiles, cleanup_ps


def _get_worker_info():
    worker_id = int(os.getenv(PARALLAX_WORKER_ID, -1))
    if worker_id == -1:
        raise RuntimeError(
            "Need to set environment variable PARALLAX_WORKER_ID")
    num_workers = int(os.getenv(PARALLAX_NUM_WORKERS, 0))
    if num_workers == 0:
        raise RuntimeError(
            "Need to set environment variable PARALLAX_NUM_WORKERS")
    return worker_id, num_workers


def parallax_run_ps(single_gpu_meta_graph_def, config,
                    export_graph=True):
    worker_id, num_workers = _get_worker_info()
    num_replicas_per_worker = len(config.resource_info['worker'][worker_id]['gpus'])

    parallax_log.debug("Launching server on worker %d" % worker_id)
    cluster_spec = get_tf_clusterspec(config.resource_info)
    server = tf.train.Server(cluster_spec, job_name='worker',
                             task_index=worker_id, protocol=config.communication_config.ps_config.protocol)
    session_target = server.target
    parallax_log.debug("Finished launching server on worker %d" % worker_id)

    ps_meta_graph_def, tensor_or_op_name_to_replica_names = graph_transform_ps(
        single_gpu_meta_graph_def,
        worker_id,
        config)

    with tf.Graph().as_default() as graph_to_run:
        parallax_log.debug("Importing PS graph on worker %d" % worker_id)
        tf.train.import_meta_graph(ps_meta_graph_def)
        if export_graph:
            export_ps_meta_graph(worker_id)

        replicated_var_init_op = None
        try:
            replicated_var_init_op = \
                tf.get_default_graph().get_operation_by_name(
                    graph_transform_lib.MIRROR_VARIABLE_INIT_OP)
        except KeyError:
            pass

        ckpt_hooks = build_ckpt_hooks(config.get_ckpt_config())
        sess_config = config.sess_config
        if sess_config is None:
            sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.train.MonitoredTrainingSession(
                master=session_target,
                is_chief=(worker_id == 0),
                checkpoint_dir=config.get_ckpt_config().ckpt_dir if worker_id == 0 else None,
                # TODO: Allow user-defined hooks
                hooks=None,
                chief_only_hooks=ckpt_hooks,
                save_checkpoint_secs=None,
                save_summaries_steps=None,
                save_summaries_secs=None,
                config=sess_config)

        parallax_log.debug(
            "Created MonitoredTrainingSession for worker %d on %s"
            % (worker_id, session_target))

        if replicated_var_init_op is not None:
            sess.run(replicated_var_init_op)

        parallax_log.debug(
            "Finished initialization process, start training on worker %d"
            % worker_id)

        step = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0])
        sess_context = ParallaxSessionContext(step,
                                              config.profile_config.profile_dir,
                                              config.profile_config.profile_steps,
                                              tensor_or_op_name_to_replica_names,
                                              num_replicas_per_worker)
        sess_context.set_parallax_session_context()
        return sess, num_workers, worker_id, num_replicas_per_worker


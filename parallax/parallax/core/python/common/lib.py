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

import logging
import os
import re
import socket
import subprocess
import uuid
import ephemeral_port_reserve

import tensorflow as tf
from tensorflow.python.ops import script_ops
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import op_def_pb2

from parallax.core.python.common.consts import *

try:
    from termcolor import colored
except:
    def colored(msg, color):
        return '\033[31m' + msg + '\033[0m'


def build_ckpt_hooks(ckpt_config):
    ckpt_dir = ckpt_config.ckpt_dir
    save_ckpt_steps = ckpt_config.save_ckpt_steps
    save_ckpt_secs = ckpt_config.save_ckpt_secs
    if ckpt_dir is None:
        return None
    elif save_ckpt_steps is None and save_ckpt_secs is None:
        return None

    # Default saver in default scaffold keeps 1,000,000 recent checkpoint files.
    saver = tf.train.Saver(tf.global_variables(), save_relative_paths=False,
                           allow_empty=True, max_to_keep=1000000)
    tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
    scaffold = tf.train.Scaffold(saver=saver)
    ckpt_hook = tf.train.CheckpointSaverHook(ckpt_dir,
                                             save_steps=save_ckpt_steps,
                                             save_secs=save_ckpt_secs,
                                             scaffold=scaffold)
    return [ckpt_hook]

parallax_log = logging.getLogger('PARALLAX')
try:
    parallax_log.setLevel(int(os.environ["PARALLAX_LOG_LEVEL"]))
except:
    pass
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter('%(levelname)s:%(thread)d:%(name)s:%(message)s', None))

parallax_log.addHandler(handler)


def remote_copy(remote_machine, local_path, remote_path, port=22):
    cmd = 'ssh -p %d %s "mkdir -p %s"' % (port, remote_machine, remote_path)
    parallax_log.warning(colored('\n$ %s' % cmd, 'red'))
    os.system(cmd)
    cmd = 'scp -P %d %s %s:%s' % (port, local_path, remote_machine, remote_path)
    parallax_log.warning(colored('\n$ %s' % cmd, 'red'))
    os.system(cmd)


def remote_exec(bash_script,
                remote_machine,
                stdout=None,
                stderr=None,
                env={},
                python_venv=None,
                port=22):
    full_cmd = ' '.join(
        map(lambda (k, v): 'export %s=%s;' % (k, v), env.iteritems()))
    if python_venv is not None:
        full_cmd += ' source %s/bin/activate; ' % python_venv
    full_cmd += bash_script

    remote_cmd = 'ssh -tt -p %d %s \'bash -c "%s"\' </dev/null' % (
    port, remote_machine, full_cmd)

    parallax_log.warning(colored('\n$ %s' % remote_cmd, 'red'))
    proc = subprocess.Popen(args=remote_cmd, shell=True, stdout=stdout,
                            stderr=stderr, preexec_fn=os.setsid)
    return proc


def _get_available_gpus(hostname):
    result = subprocess.check_output('ssh %s ls /proc/driver/nvidia/gpus' % hostname, shell=True)
    return list(range(len(result.strip().split('\n'))))


def _get_empty_port(hostname, num_ports):
    try:
        python_venv = os.environ['VIRTUAL_ENV']
    except:
        python_venv = None

    ports = []
    for i in range(num_ports):
        proc = remote_exec('python -m ephemeral_port_reserve', hostname, stdout=subprocess.PIPE, python_venv=python_venv)
        port = int(proc.stdout.readline())
        proc.wait()
        ports.append(port)
    return ports


def _parse_machine_info(machine_str):
    hostname_gpus = machine_str.split(':')
    if len(hostname_gpus) > 0:
        hostname = hostname_gpus[0]
    else:
        return []  # empty line

    if len(hostname_gpus) == 1:
        gpus = _get_available_gpus(hostname)
    elif len(hostname_gpus) == 2:
        gpus = [int(gpu) for gpu in hostname_gpus[1].split(',')]

    return [(hostname, gpus)]  # FIXME support in-graph and between-graph auto parallel


def parse_resource_info(path):
    machines = []
    with open(path) as file:
        for machine_info in file:
            machines.extend(_parse_machine_info(machine_info.strip()))

    ps = [{'hostname': hostname, 'port': _get_empty_port(hostname, 1), 'gpus': []} for hostname, _ in machines]
    worker = [{'hostname': hostname, 'port': _get_empty_port(hostname, 1 if len(gpus) == 0 else len(gpus)), 'gpus': gpus} for hostname, gpus in machines]

    resource_info = {'ps': ps, 'worker': worker}
    parallax_log.info(resource_info)
    return resource_info


def serialize_resource_info(resource_info):
    def serialize_machine(m):
        return '%s:%s:%s' % (m['hostname'], ','.join([str(port) for port in m['port']]), ','.join([str(gpu) for gpu in m['gpus']]))
    def serialize_machines(machines):
        return '+'.join([serialize_machine(m) for m in machines])
    return '^'.join(['%s_%s' % (type, serialize_machines(machines)) for type, machines in resource_info.iteritems()])


def deserialize_resource_info(resource_info_serialized):
    def deserialize_list(list):
        if len(list) == 0:
            return []
        return [int(g) for g in list.split(',')]
    def deserialize_machine(m):
      hostname, ports, gpus = m.split(':')
      return {'hostname': hostname, 'port': deserialize_list(ports), 'gpus': deserialize_list(gpus)}
    def deserialize_machines(machines):
        return [deserialize_machine(m) for m in machines.split('+')]
    resource_info = {}
    type_machines = resource_info_serialized.strip().split('^')
    for type_machine in type_machines:
        type, machines = type_machine.split('_')
        resource_info[type] = deserialize_machines(machines)
    parallax_log.info(resource_info)
    return resource_info


def get_cluster_str_for_hosts(hosts, with_slots):
    if with_slots:
        return ','.join(
            map(lambda host: '%s:%d' % (host['hostname'], len(host['gpus'])),
                hosts))
    else:
        host_list = []
        for host in hosts:
            for port in host['port']:
              new_host = {'hostname': host['hostname'], 'port': port}
              host_list.append(new_host)
        return ','.join(
            map(lambda host: '%s:%d' % (host['hostname'], host['port']), host_list))


def send_execution_time(master, worker_id, exec_time):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        parallax_log.debug('worker %d tries to connect' % worker_id)
        s.connect((master['hostname'], master['port']))
        parallax_log.debug('worker %d connected' % worker_id)
        msg = 'worker_id : %d, exec_time : %d' % (worker_id, exec_time)
        sent = s.send(msg)
        parallax_log.debug(
            'worker %d sent exec time %d secs' % (worker_id, exec_time))
        if sent == 0:
            raise RuntimeError(
                "socket connection broken for worker id : %d" % worker_id)
    except:
        raise RuntimeError(
            "socket connection broken for worker id : %d" % worker_id)

def get_average_execution_time(master, num_workers):
    try:
        # create an INET, STREAMing socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # set socket option (enable re-using the port reserved by
        # ephemeral_port_reserve)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # bind the socket to the chief worker
        s.bind(('', master['port']))

        # become a server socket
        s.listen(5)

        client_list = []
        total_exec_time = 0

        while len(client_list) != num_workers:
            connection, address = s.accept()
            client_list.append(connection)
        parallax_log.debug('number of connected clients : %d' % len(client_list))

        parallax_log.debug('all clients are connected')

        message_sent_clients = []
        while len(message_sent_clients) != num_workers:
            for connection in client_list:
                if connection in message_sent_clients:
                    continue
                buf = connection.recv(64)
                if len(buf) > 0 and 'worker' in buf:
                    nums = re.findall(r'\d+', buf)
                    worker_id = int(nums[0])
                    exec_time = int(nums[1])
                    parallax_log.debug(
                        "get message from worker %d - execution time : %d secs"
                        % (worker_id, exec_time))
                    total_exec_time += exec_time
                    message_sent_clients.append(connection)

        for connection in client_list:
            connection.close()
        s.close()
    except:
        raise RuntimeError(
            "socket connection is broken")      
    return total_exec_time / num_workers


def export_mpi_meta_graph(worker_id):
    _export_meta_graph(worker_id, 'mpi', 'MPI')

def export_ps_meta_graph(worker_id):
    _export_meta_graph(worker_id, 'ps', 'PS')

def export_hybrid_meta_graph(worker_id):
    _export_meta_graph(worker_id, 'hybrid', 'HYBRID')    
 
def _export_meta_graph(worker_id, dir, tag):
    export_meta_graph_path = \
        os.path.join(REMOTE_PARALLAX_ROOT, dir,
                     'worker-%d-%s' % (worker_id, str(uuid.uuid4())))
    parallax_log.debug("Exporting %s graph of worker %d to %s"
                      % (tag, worker_id, export_meta_graph_path))
    tf.train.export_meta_graph(export_meta_graph_path, as_text=True)


def get_tf_clusterspec(resource_info):
    tf_cluster_dict = {}
    for job in ['ps', 'worker']:
        if job not in resource_info:
            continue
        hosts = resource_info[job]
        tf_cluster_dict[job] = []
        for host in hosts:
            tf_cluster_dict[job].append(
                '%s:%d' % (host['hostname'], host['port'][0]))
    cluster_spec = tf.train.ClusterSpec(tf_cluster_dict)
    return cluster_spec

class TensorOrOpNameToReplicaNames(object):
    def __init__(self, op_defs):
        self._mapping = {}
        self._op_defs = {}
        for op_def in op_defs.op:
            self._op_defs[op_def.name] = op_def

    def _len_node_outputs(self, node_def):
        assert isinstance(node_def, node_def_pb2.NodeDef), 'node_def type %s' % type(node_def)
        op_def = self._op_defs[node_def.op]
        len_outputs = 0
        for output_argdef in op_def.output_arg:
            if output_argdef.number_attr:
                # A sequence of tensors with the same type
                len_outputs += node_def.attr[output_argdef.number_attr].i
            elif output_argdef.type_list_attr:
                # A sequence of tensors
                len_outputs += len(node_def.attr[output_argdef.type_list_attr].list.type)
            else:
                # A single tensor
                len_outputs += 1

        return len_outputs

    def update_mapping_from_tensor(self, single_tensor, replica_tensor):
        assert single_tensor.name in self._mapping
        assert single_tensor.op.name in self._mapping
        self._mapping[single_tensor.name] = [replica_tensor.name]
        self._mapping[single_tensor.op.name] = [replica_tensor.op.name]

    def extend_mapping_from_nodedef(self, single_nodedef, replica_nodedef):
        assert isinstance(single_nodedef, node_def_pb2.NodeDef), \
                'single nodedef type is %s' % type(single_nodedef)
        assert isinstance(replica_nodedef, node_def_pb2.NodeDef), \
                'replica nodedef type is %s' % type(replica_nodedef)

        def _append_mapping(tensor_or_op_name, replica_name):
            if tensor_or_op_name not in self._mapping:
                self._mapping[tensor_or_op_name] = []
            assert isinstance(self._mapping[tensor_or_op_name], list)
            self._mapping[tensor_or_op_name].append(replica_name)

        _append_mapping(single_nodedef.name, replica_nodedef.name)

        for i in range(self._len_node_outputs(single_nodedef)):
            single_tensor_name = '%s:%d' % (single_nodedef.name, i)
            replica_tensor_name = '%s:%d' % (replica_nodedef.name, i)
            _append_mapping(single_tensor_name, replica_tensor_name)

    def export(self):
        return self._mapping

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

import sys
import time
import numpy as np

import tensorflow as tf
from tensorflow.contrib.training.python.training import device_setter
from tensorflow.core.framework import attr_value_pb2

from parallax.core.python.common.graph_transform_lib import BINARY_ENCODED_COLOCATION_PREFIX
from parallax.core.python.common.graph_transform_lib import replicate_variables_to_devices
from parallax.core.python.common.graph_transform_lib import add_sync_op
from parallax.core.python.common.graph_transform_lib import update_shard_values_for_worker
from parallax.core.python.common.lib import *

def byte_size_load_fn(op):
    """Load function that computes the byte size of a single-output `Operation`.
    This is intended to be used with TRAINABLE_VARIABLES, which may affect
    communication time between parameter servers and workers. The load of other
    type operations is assumed as zero.
    Args:
        op: An `Operation` with a single output, typically a "Variable" op.
    Returns:
        The number of bytes in the output `Tensor`.
    Raises:
        ValueError: if `op` does not have a single output, or if the shape of the
            single output is not fully-defined.
    """

    if op.type == "Variable" or op.type == "VariableV2":
        return device_setter.byte_size_load_fn(op)
    return 0


class GreedyLoadBalancingStrategy(object):
    def __init__(self, num_tasks):
        """Create a new `LoadBalancingStrategy`.
        This only consider trainable variables as load of ps.
        Args:
            num_tasks: Number of ps tasks to cycle among.
        """
        self._num_tasks = num_tasks
        self._ps_loads = np.zeros(num_tasks)

    def __call__(self, op):
        """Choose a ps task index for the given `Operation`.
        Args:
            op: A `Operation` to be placed on ps.
        Returns:
            The next ps task index to use for the `Operation`. Greedily
            places the op on the least-loaded ps task so far, as determined
            by the load function.
        """
        task = np.argmin(self._ps_loads)
        self._ps_loads[task] += byte_size_load_fn(op)
        return task


def between_graph_auto_parallel_compute(meta_graph_def,
                                        worker_id,
                                        ps_device,
                                        worker_device,
                                        merge_devices,
                                        cluster_spec,
                                        config,
                                        op_library_path,
                                        num_replicas_per_worker,
                                        tensor_or_op_name_to_replica_names):
    """Returns graph replicas. This is for between-graph replication.

    Args:
        meta_graph_def: Target graph def proto for replicas.
        ps_device: Device of the ps job. If empty no ps job is used.
            Defaults to ps.
        worker_device: Device of the worker job. If empty no worker job is used.
        merge_devices: If True, merges or only sets a device
            if the device constraint is completely unset.
            merges device specification rather than overriding them.
        cluster_spec: tf.train.ClusterSpec
        op_library_path: The op library path for custom operators
        num_replicas_per_worker : Number of gpus per worker.
    """
    parallax_log.debug("BetweenGraphAutoParallelOpKernel: start on worker %d"
                      % worker_id)
    start_time = time.time()

    cluster_spec = cluster_spec.as_dict()
    ps_job_name = tf.DeviceSpec.from_string(ps_device).job
    worker_job_name = tf.DeviceSpec.from_string(worker_device).job
    num_workers = len(cluster_spec[worker_job_name])
    num_pss = len(cluster_spec[ps_job_name])

    # TODO: Partitioning large variables can be helpful for load balancing.
    ps_strategy = GreedyLoadBalancingStrategy(num_pss)

    if op_library_path is not None:
        tf.load_op_library(op_library_path)

    with tf.Graph().as_default() as transformed_graph:
        with tf.device(
                tf.train.replica_device_setter(
                    ps_device=ps_device,
                    worker_device=worker_device,
                    merge_devices=merge_devices,
                    cluster=cluster_spec,
                    ps_strategy=ps_strategy)):
            import_start_time = time.time()
            tf.train.import_meta_graph(meta_graph_def)
            import_duration = time.time() - import_start_time
            parallax_log.debug("Time to import meta graph : %.3f seconds"
                              % import_duration)

            master_var_op_to_mirror_vars = None
            if config.communication_config.ps_config.replicate_variables:
                if not config.sync:
                    raise ValueError(
                        'replicated option is only possible with sync')
                master_var_op_to_mirror_vars = \
                    replicate_variables_to_devices(
                        meta_graph_def, worker_device,
                        num_replicas_per_worker)

            if config.sync:
                add_sync_start_time = time.time()
                add_sync_op(worker_id, num_workers,
                             master_var_op_to_mirror_vars, ps_device,
                             worker_device, config.average_sparse,
                             tensor_or_op_name_to_replica_names)
                add_sync_duration = time.time() - add_sync_start_time
                parallax_log.debug("Time to add sync operation : %.3f seconds"
                                  % add_sync_duration)
                sys.stdout.flush()

        for op in tf.get_default_graph().get_operations():
            op_to_bind_to_device = None
            for colocation_group in op.colocation_groups():
                assert colocation_group.startswith(\
                    BINARY_ENCODED_COLOCATION_PREFIX)
                op_name_to_bind_to = \
                     colocation_group[len(BINARY_ENCODED_COLOCATION_PREFIX):]\
                         .decode("ascii")
                op_to_bind_to = tf.get_default_graph()\
                    .get_operation_by_name(op_name_to_bind_to)
                # Apply colocation constraints explicitly
                if op_to_bind_to == op:
                    continue
                if op_to_bind_to_device is None:
                    op_to_bind_to_device = op_to_bind_to.device
                else:
                    if op_to_bind_to_device != op_to_bind_to.device:
                        # Folllow worker device if devices are conflicted.
                        if 'ps' in op_to_bind_to_device:
                            assert 'ps' not in op_to_bind_to.device
                            op_to_bind_to_device = op_to_bind_to.device
                        else:
                            assert 'ps' in op_to_bind_to.device
                op_to_bind_to_device = op_to_bind_to.device

            if op_to_bind_to_device is not None:
                op._set_device(op_to_bind_to_device)
                new_col_groups = []
                for colocation_group in op.colocation_groups():
                    op_name_to_bind_to = colocation_group[\
                        len(BINARY_ENCODED_COLOCATION_PREFIX):].decode("ascii")
                    op_to_bind_to = tf.get_default_graph()\
                        .get_operation_by_name(op_name_to_bind_to)
                    if op_to_bind_to.device == op_to_bind_to_device:
                        new_col_groups.append(colocation_group)
                op._set_attr("_class", attr_value_pb2.AttrValue(
                    list=attr_value_pb2.AttrValue.ListValue(s=new_col_groups)))

        update_shard_values_for_worker(num_workers, worker_id)

        export_start_time = time.time()
        transformed_meta_graph_def = \
            tf.train.export_meta_graph('graph_worker_%d' % worker_id,
                                       as_text=True)
        export_duration = time.time() - export_start_time
        parallax_log.debug(
            "Time to export meta graph : %.3f seconds" % export_duration)

    duration = time.time() - start_time
    parallax_log.debug(
        "BetweenGraphAutoParallelOpKernel: end (took %.3f seconds)" % duration)
    return transformed_meta_graph_def


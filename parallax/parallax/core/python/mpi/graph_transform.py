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

import tensorflow as tf
import horovod.tensorflow as hvd

from parallax.core.python.common.graph_transform_lib import get_all_control_consumers
from parallax.core.python.common.graph_transform_lib import update_consumers
from parallax.core.python.common.graph_transform_lib import update_control_consumers
from parallax.core.python.common.graph_transform_lib import update_shard_values_for_worker
from parallax.core.python.common.lib import *


def _add_broadcast_ops():
    global_step = tf.identity(tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0].op.outputs[0])
    bcast_global_variables_ops = []
    for var in tf.global_variables():
        bcast_global_variables_ops.append(
            tf.assign(var, hvd.broadcast(var, global_step, 0)))
    with tf.control_dependencies(bcast_global_variables_ops):
        tf.no_op(name='auto_parallel_bcast_global_vars')


def _add_aggregation_ops(gradients_info, op_to_control_consumer_ops, config):
    global_step = tf.identity(tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0].op.outputs[0])
    grad_tensor = gradients_info._grad
    if isinstance(grad_tensor, tf.Tensor):
        grad = grad_tensor
        grad_consumers = [c for c in grad.consumers()]
        agg_grad = hvd.allreduce(grad,
                                 global_step,
                                 average_dense=True,
                                 average_sparse=config.average_sparse,
                                 use_allgatherv=config.communication_config.mpi_config.use_allgatherv)
        update_consumers(grad_consumers, grad, agg_grad)
        update_control_consumers(op_to_control_consumer_ops[grad.op],
                                 grad.op, agg_grad.op)
    else:
        grad = grad_tensor.values
        indices = grad_tensor.indices
        dense_shape = grad_tensor.dense_shape
        grad_consumers = [c for c in grad.consumers()]
        indices_consumers = [c for c in indices.consumers()]
        agg_grad = \
            hvd.allreduce(tf.IndexedSlices(grad, indices, dense_shape),
                          global_step,
                          average_dense=True,
                          average_sparse=config.average_sparse,
                          use_allgatherv=config.communication_config.mpi_config.use_allgatherv)
        update_consumers(grad_consumers, grad, agg_grad.values)
        update_consumers(indices_consumers, indices, agg_grad.indices)
        update_control_consumers(op_to_control_consumer_ops[grad.op],
                                 grad.op, agg_grad.values.op)
        update_control_consumers(
            op_to_control_consumer_ops[indices.op], indices.op,
            agg_grad.indices.op)
    gradients_info._grad = agg_grad


def graph_transform_mpi(single_gpu_meta_graph_def, config,
                        op_library_path=None):
    if op_library_path is not None:
        tf.load_op_library(op_library_path)

    with tf.Graph().as_default() as replica:
        tf.train.import_meta_graph(single_gpu_meta_graph_def)

        tensor_or_op_name_to_replica_names = {}
        for op in replica.get_operations():
            tensor_or_op_name_to_replica_names[op.name] = [op.name]
            for output in op.outputs:
                tensor_or_op_name_to_replica_names[output.name] = [output.name]

        # Initialize horovod
        hvd.init()

        num_workers = hvd.size()
        worker_id = hvd.rank()
        update_shard_values_for_worker(num_workers, worker_id)

        op_to_control_consumer_ops = get_all_control_consumers(replica)
        trainable_variable_ops = [var.op for var in tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)]

        for gradients_info in tf.get_collection(tf.GraphKeys.GRADIENTS_INFO):
            target_tensor = gradients_info._target
            if target_tensor.op not in trainable_variable_ops:
                parallax_log.debug(
                    "Gradient for non-trainable variable %s is created, ignore"
                    % target_tensor.op.name)
                continue

            _add_aggregation_ops(gradients_info, op_to_control_consumer_ops, config)
        _add_broadcast_ops()

    return tf.train.export_meta_graph(graph=replica), \
           tensor_or_op_name_to_replica_names

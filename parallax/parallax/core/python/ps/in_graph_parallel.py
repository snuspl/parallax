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

import time

import tensorflow as tf
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import ops

from parallax.core.python.common.graph_transform_lib import *
from parallax.core.python.common.lib import *


def _get_ops_to_replicate(gradiend_info_list):
    grads = [gradient_info._grad for gradient_info in gradiend_info_list]

    grad_related = set()
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            grad_related.add(grad.indices)
            grad_related.add(grad.values)
            grad_related.add(grad.dense_shape)
        elif isinstance(grad, tf.Tensor):
            grad_related.add(grad)
        else:
            raise RuntimeError("Incorrect grad.")

    grads_ancestor_ops = get_ancestors([grad.op for grad in grad_related],
                                       include_control_inputs=True)
    pipeline_ops = get_pipeline_ops(grads_ancestor_ops)

    global_var_related_ops = set()
    for global_var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        global_var_related_ops.add(global_var.op)
        global_var_related_ops.add(global_var.initializer)
        global_var_related_ops.add(global_var._snapshot.op)

    table_related_ops = set()
    for table_init in tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS):
        table_related_ops.add(table_init)
        table_related_ops.add(table_init.inputs[0].op)

    # Assume that all variables are member of either GLOBAL_VARIABLES
    # or LOCAL_VARIABLES.
    local_var_op_to_var = \
        dict([(var.op, var)
              for var in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)])
    local_var_ops = set(local_var_op_to_var.keys())
    local_var_ops.intersection_update(grads_ancestor_ops)

    ops_to_replicate = set()
    ops_to_replicate.update(grads_ancestor_ops)
    ops_to_replicate.update(pipeline_ops)
    ops_to_replicate.difference_update(global_var_related_ops)
    ops_to_replicate.difference_update(table_related_ops)
    ops_to_replicate.update(
        [local_var_op_to_var[var_op].initializer for var_op in local_var_ops])

    return ops_to_replicate


def _get_multi_gpu_meta_graph(single_gpu_meta_graph_def, op_names_to_replicate,
                              op_names_to_share, num_replicas,
                              tensor_or_op_name_to_replica_names):
    multi_gpu_graph_def = \
        construct_multi_gpu_graph_def(
            single_gpu_meta_graph_def.graph_def,
            op_names_to_replicate,
            op_names_to_share,
            num_replicas,
            tensor_or_op_name_to_replica_names)
    multi_gpu_meta_graph_def = meta_graph_pb2.MetaGraphDef()
    multi_gpu_meta_graph_def.CopyFrom(single_gpu_meta_graph_def)
    multi_gpu_meta_graph_def.graph_def.Clear()
    multi_gpu_meta_graph_def.graph_def.CopyFrom(multi_gpu_graph_def)
    return multi_gpu_meta_graph_def


def _handle_collection_def(multi_gpu_meta_graph_def, op_names_to_replicate,
                           num_replicas):
    allow_bytes_list_keys = [tf.GraphKeys.QUEUE_RUNNERS,
                             tf.GraphKeys.GLOBAL_VARIABLES,
                             tf.GraphKeys.TRAINABLE_VARIABLES,
                             tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
                             tf.GraphKeys.LOCAL_VARIABLES,
                             tf.GraphKeys.MODEL_VARIABLES,
                             tf.GraphKeys.GRADIENTS_INFO,
                             tf.GraphKeys.GLOBAL_STEP]
    keys_to_remove = []
    for key, col_def in multi_gpu_meta_graph_def.collection_def.items():
        kind = col_def.WhichOneof("kind")
        # Update node_list collections (e.g., GLOBAL_STEP, TRAIN_OP, UPDATE_OP,
        # LOSSES, ...)
        if kind == 'node_list':
            new_col_def = get_new_col_def_of_node_list(
                            col_def, op_names_to_replicate, num_replicas)
            multi_gpu_meta_graph_def.collection_def[key].Clear()
            multi_gpu_meta_graph_def.collection_def[key].CopyFrom(new_col_def)
        elif kind == 'bytes_list':
            if ops.get_from_proto_function(key):
                # Collections in allow_bytes_list_keys will be handled
                # explicitly below
                # (e.g., QUEUE_RUNNERS, LOCAL_VARIABLES, ...)
                if key in allow_bytes_list_keys:
                    continue
                # Remove unhandled collections (e.g., COND_CONTEXT)
                # TODO: Handle all protos in tf.GraphKeys
                else:
                    keys_to_remove.append(key)
            # Keep collections without proto function
            # (e.g., user defined string)
            else:
                continue
        else:
            raise RuntimeError("Should not reach here")
    for key in keys_to_remove:
        del multi_gpu_meta_graph_def.collection_def[key]

    # Update QUEUE_RUNNERS and LOCAL_VARIABLES collection
    update_queue_runners(multi_gpu_meta_graph_def, op_names_to_replicate,
                          num_replicas)
    update_local_variables(multi_gpu_meta_graph_def, op_names_to_replicate,
                            num_replicas)
    update_shard_info_for_in_graph(multi_gpu_meta_graph_def, num_replicas)


def in_graph_auto_parallel_compute(single_gpu_meta_graph_def,
                                   num_replicas,
                                   config,
                                   op_library_path,
                                   tensor_or_op_name_to_replica_names):
    """Returns a graph replica. This is for in-graph replication.

  Args:
    single_gpu_meta_graph_def: Target meta graph definition proto for replicas.
    num_replicas: shape {1}. Number of replicas, gpus are utilized
        as many as num_replicas.
    average_sparse: shape {1}. Whether to average sparse values or not.
  Returns:
    A tensor which contains serialized meta-graph def proto after replication.
    The output tensor is converted into numpy array.
  """
    parallax_log.debug("InGraphAutoParallelOpKernel: start")
    start_time = time.time()

    if op_library_path is not None:
        tf.load_op_library(op_library_path)

    average_option = SPARSE_AVERAGE_BY_COUNTER\
        if config.average_sparse else SPARSE_NO_AVERAGE

    with tf.Graph().as_default():
        import_start_time = time.time()
        tf.train.import_meta_graph(single_gpu_meta_graph_def)
        import_duration = time.time() - import_start_time
        parallax_log.debug(
            "Time to import single-GPU meta graph : %.3f seconds"
            % import_duration)

        gradient_info_list = \
            [gi for gi in tf.get_collection(tf.GraphKeys.GRADIENTS_INFO)]

        ops_to_replicate = _get_ops_to_replicate(gradient_info_list)
        op_names_to_replicate = set([op.name for op in ops_to_replicate])
        ops_to_share = set(tf.get_default_graph().get_operations())
        ops_to_share.difference_update(ops_to_replicate)
        op_names_to_share = set([op.name for op in ops_to_share])
        op_to_control_consumer_ops = \
            get_all_control_consumers(tf.get_default_graph())

    multi_gpu_meta_graph_def = \
        _get_multi_gpu_meta_graph(single_gpu_meta_graph_def,
                                  op_names_to_replicate, op_names_to_share,
                                  num_replicas,
                                  tensor_or_op_name_to_replica_names)
    _handle_collection_def(multi_gpu_meta_graph_def, op_names_to_replicate,
                           num_replicas)

    # Delete GRADIENTS_INFO collection temporarily
    del multi_gpu_meta_graph_def.collection_def[tf.GraphKeys.GRADIENTS_INFO]

    multi_gpu_meta_graph_def = add_aggregate_gradients_ops(
        multi_gpu_meta_graph_def,
        op_names_to_replicate,
        op_to_control_consumer_ops,
        gradient_info_list,
        num_replicas,
        average_option)

    duration = time.time() - start_time
    parallax_log.debug(
        "InGraphAutoParallelOpKernel: end (took %.3f seconds)" % duration)
    return multi_gpu_meta_graph_def


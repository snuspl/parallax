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

import re
import sys
import time
from functools import reduce

import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.core.protobuf import gradients_info_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import queue_runner_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tensorflow as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.gradients_impl import GradientsInfo
from tensorflow.python.training import queue_runner
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import session_manager
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import compat

import parallax
from parallax.core.python.common.lib import *
from parallax.core.python.common import shard

MAX_INT64 = int(2 ** 63 - 1)

dense_var_update_op_types = {"ApplyGradientDescent": 2,
                             "ApplyProximalGradientDescent": 4,
                             "ApplyAdadelta": 6,
                             "ApplyAdagrad": 3,
                             "ApplyProximalAdagrad": 5,
                             "ApplyAdagradDA": 3,
                             "ApplyFtrl": 3,
                             "ApplyMomentum": 3,
                             "ApplyAdam": 9,
                             "ApplyRMSProp": 7,
                             "ApplyCenteredRMSProp": 8,
                             "AssignAdd": 1,
                             "AssignSub": 1}
sparse_var_update_op_types = {"ScatterUpdate": (1, 2),
                              "ScatterAdd": (1, 2),
                              "ScatterSub": (1, 2),
                              "ScatterMul": (1, 2),
                              "ScatterDiv": (1, 2),
                              # TODO: support other sparse update ops
                              "SparseApplyAdagrad": (4, 3)}

VARIABLE_OP_TYPES = [
    "Variable", "VariableV2",
    "AutoReloadVariable", "VarHandleOp", "ReadVariableOp"]

STAGE_OP_TYPES = ["Stage"]
UNSTAGE_OP_TYPES = ["Unstage"]

QUEUE_OP_TYPES = [
    "RandomShuffleQueue", "RandomShuffleQueueV2",
    "FIFOQueue", "FIFOQueueV2",
    "PaddingFIFOQueue", "PaddingFIFOQueueV2",
    "PriorityQueue", "PriorityQueueV2"]
DEQUEUE_OP_TYPES = [
    "ReaderRead", "ReaderReadV2",
    "ReaderReadUpTo", "ReaderReadUpToV2",
    "ReaderRestoreState", "ReaderRestoreStateV2",
    "QueueDequeueMany", "QueueDequeueManyV2",
    "QueueDequeue", "QueueDequeueV2",
    "QueueDequeueUpTo", "QueueDequeueUpToV2"]
ITERATOR_OP_TYPES = [
    "Iterator", "IteratorV2"
]

UPDATE_OP_VAR_POS = 0
SPARSE_AVERAGE_BY_COUNTER = 1
SPARSE_NO_AVERAGE = 3
PARALLAX_PREFIX = u"AutoParallel-"
PARALLAX_REPLICA_PREFIX = u"%sReplica-" % PARALLAX_PREFIX
MIRROR_VARIABLE_INIT_OP = "auto_parallel_replicated_var_init_op"
BINARY_ENCODED_COLOCATION_PREFIX = b"loc:@"
COLOCATION_PREFIX = "loc:@"
PARALLAX_GLOBAL_GRADS = 'PARALLAX_GLOBAL_GRADS'

def parallax_replica_prefix(replica_id):
    return '%s%s' % (PARALLAX_REPLICA_PREFIX, str(replica_id))


def _get_op_name(tensor_name):
    return tensor_name.replace('^', '').split(':')[0]


def get_consumers(op):
    # get a flat list from [output[0].consumers(), output[1].consumers(), ...]
    return [consumer for output in op.outputs
            for consumer in output.consumers()]


def get_all_control_consumers(graph):
    op_to_control_consumer_ops = dict([(op, [])
                                       for op in graph.get_operations()])
    for op in graph.get_operations():
        for control_input_op in op.control_inputs:
            op_to_control_consumer_ops[control_input_op].append(op)
    return op_to_control_consumer_ops


# Be careful using op.consumers() directly as an argument,
# since this causes incorrect list iteration.
def update_consumers(consumers, old_tensor, new_tensor):
    for consumer_op in consumers:
        for i, x in enumerate(consumer_op.inputs):
            if x == old_tensor:
                consumer_op._update_input(i, new_tensor)

def update_control_consumers(control_consumer_ops, old_op, new_op):
    for control_consumer_op in control_consumer_ops:
         control_inputs = list(control_consumer_op.control_inputs)
         size = len(control_inputs)
         control_inputs.remove(old_op)
         assert size - 1 == len(control_inputs)
         control_inputs.append(new_op)
         assert size == len(control_inputs)
         control_consumer_op._remove_all_control_inputs()
         control_consumer_op._add_control_inputs(control_inputs)

# Starting from start_ops, follow the computation graph from consumer to input
# to find ancestors. Stop navigating the graph at end_ops. Include both
# start_ops and end_ops in the returning set of ancestor ops.
def get_ancestors(start_ops, end_ops=[], include_control_inputs=False):
    ancestor_ops = set()
    queue = []
    queue.extend(start_ops)

    while len(queue) > 0:
        curr_op = queue.pop()
        if curr_op in ancestor_ops:
            continue
        ancestor_ops.add(curr_op)
        if curr_op in end_ops:
            continue
        queue.extend([input.op for input in curr_op.inputs])
        consumers = get_consumers(curr_op)
        if include_control_inputs:
            queue.extend([op for op in curr_op.control_inputs])
    return ancestor_ops


def _place_post_grad_agg_ops(ps_device,
                             var_op_to_agg_grad,
                             var_op_to_apply_grad_op):
    def _find_agg_grad_descendant_ops(agg_grad_ops, apply_grad_ops):
        agg_grad_descendant_ops = set()
        queue = []
        queue.extend(agg_grad_ops)

        while len(queue) > 0:
            curr_op = queue.pop()
            if curr_op in agg_grad_descendant_ops:
                continue
            agg_grad_descendant_ops.add(curr_op)
            if curr_op in apply_grad_ops:
                continue
            curr_op_consumers = get_consumers(curr_op)
            queue.extend(curr_op_consumers)
        return agg_grad_descendant_ops

    SHARED = -1

    def _assign(op_to_task, agg_grad_ops, apply_grad_ops,
                apply_grad_ancestor_ops, ancestors_diff_descendants,
                is_parent_to_child):
        queue = []
        stop = set()
        if is_parent_to_child:
            queue.extend(agg_grad_ops)
            stop.update(apply_grad_ops)
        else:
            queue.extend(apply_grad_ops)
            stop.update(agg_grad_ops)

        visited = set()
        while len(queue) > 0:
            curr_op = queue.pop(0)
            if curr_op in visited:
                continue
            visited.add(curr_op)
            # already assigned to a task,
            # so skip computing placement and then add next ops to the queue
            if curr_op in op_to_task and curr_op not in stop:
                if is_parent_to_child:
                    # do not care about ops not required for applying gradients
                    queue.extend(
                        [consumer for consumer in get_consumers(curr_op)
                         if consumer in apply_grad_ancestor_ops])
                else:
                    queue.extend([input.op for input in curr_op.inputs])
                continue

            if is_parent_to_child:
                placement_reference_ops = \
                    set([input.op for input in curr_op.inputs])
                placement_reference_ops = placement_reference_ops.difference(
                    ancestors_diff_descendants)
            else:
                placement_reference_ops = set(get_consumers(curr_op))
                placement_reference_ops = placement_reference_ops.intersection(
                    apply_grad_ancestor_ops)

            is_ready = True
            for ref_op in placement_reference_ops:
                if ref_op not in op_to_task:
                    is_ready = False
                    break

            if is_ready:
                placement_reference_tasks = \
                    [op_to_task[ref_op] for ref_op in placement_reference_ops]
            else:
                # requeue and wait for references
                queue.append(curr_op)
                continue

            unique_tasks = set(placement_reference_tasks)
            curr_op_task = None
            if len(unique_tasks) == 0:
                raise RuntimeError(
                    "Should have placement reference for operation %s"
                    % curr_op.name)
            elif len(unique_tasks) == 1:
                curr_op_task = unique_tasks.pop()
                op_to_task[curr_op] = curr_op_task
            else:
                # priority: assigned placement > SHARED
                if SHARED in unique_tasks:
                    unique_tasks.remove(SHARED)
                if len(unique_tasks) == 1:
                    curr_op_task = unique_tasks.pop()
                    op_to_task[curr_op] = curr_op_task
                else:
                    # multiple device placement -> SHARED
                    assert len(unique_tasks) > 1
                    curr_op_task = SHARED
                    op_to_task[curr_op] = SHARED
            parallax_log.debug('post_grad_agg_op %s is assigned to ps task %d'
                              % (curr_op.name, curr_op_task))

            if curr_op_task == SHARED:
                # TODO: do not assign all SHARED ops to task 0
                # - we can do better
                curr_op_task = 0
            ps_device.task = curr_op_task
            curr_op._set_device(ps_device)

            if curr_op not in stop:
                if is_parent_to_child:
                    # do not care about ops not required for applying gradients
                    queue.extend(
                        [consumer for consumer in get_consumers(curr_op)
                         if consumer in apply_grad_ancestor_ops])
                else:
                    queue.extend([input.op for input in curr_op.inputs])

    op_to_task = {}
    agg_grad_ops = []
    for var_op, agg_grad in var_op_to_agg_grad.items():
        var_device = tf.DeviceSpec.from_string(var_op.device)
        if agg_grad[0] != None:
            agg_grad_ops.append(agg_grad[0].op)
            op_to_task[agg_grad[0].op] = var_device.task
        agg_grad_ops.append(agg_grad[1].op)
        op_to_task[agg_grad[1].op] = var_device.task

    apply_grad_ops = []
    for var_op, apply_grad_op in var_op_to_apply_grad_op.items():
        var_device = tf.DeviceSpec.from_string(var_op.device)
        apply_grad_ops.append(apply_grad_op)
        # colocate apply_grad and variable
        apply_grad_op._set_device(var_device)
        op_to_task[apply_grad_op] = var_device.task

    # Note(gyeongin): Need to include control dependency ops in ancestors and
    # descendants or not?
    apply_grad_ancestor_ops = get_ancestors(apply_grad_ops, agg_grad_ops)
    agg_grad_descendant_ops = _find_agg_grad_descendant_ops(agg_grad_ops,
                                                            apply_grad_ops)
    ancestors_diff_descendants = \
        apply_grad_ancestor_ops.difference(agg_grad_descendant_ops)
    parallax_log.debug(
        "apply_grad_ancestor_ops: %d" % len(apply_grad_ancestor_ops))
    parallax_log.debug(
        "agg_grad_descendant_ops: %d" % len(agg_grad_descendant_ops))
    parallax_log.debug(
        "ancestors diff descendants: %d" % len(ancestors_diff_descendants))
    parallax_log.debug(
        "descendants diff ancestors: %d"
        % len(agg_grad_descendant_ops.difference(apply_grad_ancestor_ops)))

    _assign(op_to_task, agg_grad_ops, apply_grad_ops, apply_grad_ancestor_ops,
            ancestors_diff_descendants, is_parent_to_child=True)
    _assign(op_to_task, agg_grad_ops, apply_grad_ops, apply_grad_ancestor_ops,
            ancestors_diff_descendants, is_parent_to_child=False)


def add_sync_op(worker_id,
                 num_workers,
                 master_var_op_to_mirror_vars,
                 ps_device,
                 worker_device,
                 average_sparse,
                 tensor_or_op_name_to_replica_names):
    """Adds additional ops needed for synchronous distributed training into
    current graph.

    Main purpose of additional ops are:
    1. Initialization
    2. Synchronization
    3. Gradient aggregation

    Args:
        worker_id: The worker id
        num_workers: Total number of workers to synchronize
        master_var_op_to_mirror_vars: The dictionary of master variable op name
            -> list of replicated variables, could be None
        worker_device : The worker device string
        average_sparse: Whether to average sparse values or not.
    Returns:
        None
    """

    def _get_accum_apply_and_agg_grad(var_op, grad, indices, dense_shape):
        if indices is None:
            grad_accum = tf.ConditionalAccumulator(
                grad.dtype,
                shape=var_op.outputs[0].get_shape(),
                shared_name=var_op.name + "/grad_accum")
            # Get a copy of consumers list before creating accum_apply_op
            grad_consumers = [c for c in grad.consumers()]
            accum_apply_op = grad_accum.apply_grad(
                grad, local_step=MAX_INT64,
                name=grad.op.name + '_accum_apply_grad')
            agg_grad = grad_accum.take_grad(num_workers,
                                            name=var_op.name + '_take_grad')
            update_consumers(grad_consumers, grad, agg_grad)
            update_control_consumers(op_to_control_consumer_ops[grad.op],
                                      grad.op, agg_grad.op)
        else:
            grad_indexed_slices = tf.IndexedSlices(values=grad, indices=indices,
                                                   dense_shape=dense_shape)
            grad_accum = tf.SparseConditionalAccumulator(
                grad.dtype,
                shape=grad.shape,
                shared_name=var_op.name + "/grad_accum")
            # Get a copy of consumers list before creating accum_apply_op
            indices_consumers = [c for c in indices.consumers()]
            grad_consumers = [c for c in grad.consumers()]
            accum_apply_op = grad_accum.apply_indexed_slices_grad(
                grad_indexed_slices, local_step=MAX_INT64,
                name=grad.op.name + '_accum_apply_grad')
            average_option = SPARSE_NO_AVERAGE
            if average_sparse:
                average_option = SPARSE_AVERAGE_BY_COUNTER
            agg_grad = grad_accum.take_indexed_slices_grad(
                num_workers, average_option=average_option,
                name=var_op.name + '_take_grad')
            agg_indices = agg_grad.indices
            if indices.dtype != agg_grad.indices.dtype:
                agg_indices = tf.cast(agg_grad.indices, indices.dtype)
            agg_grad = tf.IndexedSlices(values=agg_grad.values,
                                        indices=agg_indices,
                                        dense_shape=agg_grad.dense_shape)
            assert isinstance(agg_grad, tf.IndexedSlices)
            update_consumers(indices_consumers, indices, agg_grad.indices)
            update_consumers(grad_consumers, grad, agg_grad.values)
            update_control_consumers(op_to_control_consumer_ops[indices.op],
                                      indices.op, agg_grad.indices.op)
            update_control_consumers(op_to_control_consumer_ops[grad.op],
                                      grad.op, agg_grad.values.op)
        return accum_apply_op, agg_grad

    def _get_mirror_variable_update_ops(master_var_op_to_mirror_vars,
                                        grad_apply_finished, var):
        with tf.device(this_worker_cpu):
            with tf.control_dependencies(grad_apply_finished):
                updated_value = var.read_value()
        update_ops = []
        for mirror_var in master_var_op_to_mirror_vars[var.op]:
            with tf.device(mirror_var.device):
                update_ops.append(mirror_var.assign(updated_value))
        return update_ops

    def _replace_update_op_with_read_op(var_op, var_update_op, finish_op):
        var_update_consumers = [c for c in var_update_op.outputs[0].consumers()]
        for consumer in var_update_consumers:
            parallax_log.debug(
                'var: %s, var_update : %s, consumer : %s'
                % (var_op.name, var_update_op.name, consumer.name))
            assert consumer.type not in all_var_update_op_types

        # TODO: exploit locality: read updated value from mirror
        with tf.control_dependencies([finish_op]):
            with tf.device(var_op.device):
                updated_var_value = global_var_op_to_var[var_op].read_value()
        update_consumers(var_update_consumers, var_update_op.outputs[0],
                          updated_var_value)
        tensor_or_op_name_to_replica_names.update_mapping_from_tensor(
            var_update_op.outputs[0], updated_var_value)

    this_worker_cpu = tf.DeviceSpec.from_string(worker_device)
    this_worker_cpu.device_type = 'CPU'
    this_worker_cpu.device_index = 0
    is_chief = worker_id == 0

    trainable_var_op_to_var = \
        dict([(var.op, var)
              for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
    global_var_op_to_var = \
        dict([(var.op, var)
              for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
    op_to_control_consumer_ops = \
        get_all_control_consumers(tf.get_default_graph())

    var_op_to_agg_grad = {}
    var_op_to_accum_apply_op = {}

    # Aggregate gradients from different workers using ConditionalAccumulator.
    # var_op_to_agg_grad and var_op_to_accum_apply_op are updated.
    for gradients_info in tf.get_collection(tf.GraphKeys.GRADIENTS_INFO):
        target_tensor = gradients_info._target
        if target_tensor.op not in trainable_var_op_to_var:
            parallax_log.debug(
                "Gradient for non-trainable variable %s is created, "
                "do not insert accumulator for aggregating this gradient"
                % target_tensor.op.name)
            continue
        var_op = target_tensor.op
        grad_tensor = gradients_info._grad
        if isinstance(grad_tensor, tf.Tensor):
            grad = grad_tensor
            indices = None
            dense_shape = None
        else:
            grad = grad_tensor.values
            indices = grad_tensor.indices
            dense_shape = grad_tensor.dense_shape
        with tf.device(var_op.device), tf.name_scope(""):
            accum_apply_op, agg_grad = \
                _get_accum_apply_and_agg_grad(var_op, grad, indices,
                                              dense_shape)
        gradients_info._grad = agg_grad
        if indices == None:
            var_op_to_agg_grad[var_op] = (None, agg_grad)
        else:
            var_op_to_agg_grad[var_op] = (agg_grad.indices, agg_grad.values)
        var_op_to_accum_apply_op[var_op] = accum_apply_op

    global_step_op = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0].op
    assert len(tf.get_collection(tf.GraphKeys.GLOBAL_STEP)) == 1

    var_op_to_finish_op = {}
    trainable_var_op_to_update_op = {}
    non_trainable_var_op_to_update_op = {}
    all_var_update_op_types = list(dense_var_update_op_types.keys()) \
                              + list(sparse_var_update_op_types.keys())
    for op in tf.get_default_graph().get_operations():
        # Find variable update ops
        if not op.type in all_var_update_op_types:
            continue

        var_update_op = op
        var_op = var_update_op.inputs[UPDATE_OP_VAR_POS].op
        if var_op not in global_var_op_to_var \
                or var_update_op == global_var_op_to_var[var_op].initializer:
            continue

        assert var_op not in trainable_var_op_to_update_op
        assert var_op not in non_trainable_var_op_to_update_op

        if var_op in trainable_var_op_to_var:
            trainable_var_op_to_update_op[var_op] = var_update_op
            is_trainable = True
        else:
            non_trainable_var_op_to_update_op[var_op] = var_update_op
            is_trainable = False

        with tf.device(var_op.device), tf.name_scope(""):
            var_update_sync_queues = \
                [tf.FIFOQueue(1, [tf.bool], shapes=[[]],
                              name='auto_parallel_%s_update_sync_queue_%d'
                                   % (var_op.name, i),
                              shared_name='auto_parallel_%s'
                                          '_update_sync_queue_%d'
                                          % (var_op.name, i))
                 for i in range(num_workers)]

            queue_ops = []
            if is_chief:
                if is_trainable:
                    var_update_deps = \
                        [var_op_to_accum_apply_op[var_op], var_update_op]
                else:
                    var_update_deps = [var_update_op]
                # Chief enqueues tokens to all other workers
                # after executing variable update
                token = tf.constant(False)
                with tf.control_dependencies(var_update_deps):
                   for i, q in enumerate(var_update_sync_queues):
                        if i != worker_id:
                            def true_fn():
                                with tf.control_dependencies([q.dequeue()]):
                                    return q.enqueue(token)
                            queue_ops.append(tf.cond(q.size() > 0, true_fn, lambda: q.enqueue(token)))
                        else:
                            queue_ops.append(tf.no_op())
            else:
                # wait for execution of var_update_op
                if is_trainable:
                    with tf.control_dependencies(
                            [var_op_to_accum_apply_op[var_op]]):
                        dequeue = var_update_sync_queues[worker_id].dequeue()
                else:
                    dequeue = var_update_sync_queues[worker_id].dequeue()
                queue_ops.append(dequeue)

            # Only dense trainable variables are replicated locally
            if master_var_op_to_mirror_vars is not None \
                    and var_op in master_var_op_to_mirror_vars:
                mirror_variable_update_ops = _get_mirror_variable_update_ops(
                    master_var_op_to_mirror_vars,
                    queue_ops,
                    trainable_var_op_to_var[var_op])
                with tf.device(this_worker_cpu):
                    finish_op = tf.group(*mirror_variable_update_ops)
            else:
                finish_op = tf.group(*queue_ops)

            # Exceptional case: add additional dependencies for global_step
            if var_op == global_step_op and not is_chief:
                # Chief worker's finish_op already has update_op
                # as control input
                deps = [finish_op]
                deps.extend([inp.op for inp in var_update_op.inputs])
                deps.extend([inp for inp in var_update_op.control_inputs])
                finish_op = tf.group(*deps)
            var_op_to_finish_op[var_op] = finish_op

    # Place computation ops of aggregated gradients on PS
    _place_post_grad_agg_ops(tf.DeviceSpec.from_string(ps_device),
                             var_op_to_agg_grad, trainable_var_op_to_update_op)

    # Replace variable update op with finish_op (control input)
    # or read_op (input)
    for var_op, finish_op in var_op_to_finish_op.items():
        if var_op in trainable_var_op_to_update_op:
            var_update_op = trainable_var_op_to_update_op[var_op]
        else:
            var_update_op = non_trainable_var_op_to_update_op[var_op]
        update_control_consumers(op_to_control_consumer_ops[var_update_op],
                                  var_update_op, finish_op)
        _replace_update_op_with_read_op(var_op, var_update_op, finish_op)

def replicate_variables_to_devices(meta_graph_def,
                                    worker_device,
                                    num_replicas_per_worker):
    var_op_name_to_original_device_str = {}
    for node in meta_graph_def.graph_def.node:
        if 'Variable' in node.op:
            var_op_name_to_original_device_str[node.name] = node.device

    sparse_var_op_names = []
    for gradients_info in tf.get_collection(tf.GraphKeys.GRADIENTS_INFO):
        grad_tensor = gradients_info._grad
        if not isinstance(grad_tensor, tf.Tensor):
            target_tensor = gradients_info._target
            sparse_var_op_names.append(target_tensor.op.name)

    worker_device = tf.DeviceSpec.from_string(worker_device)

    master_var_op_to_mirror_vars = {}
    mirror_variable_init_ops = []

    for master_var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        # Do not replicate sparse variables
        if master_var.op.name in sparse_var_op_names:
            continue
        master_var_op_to_mirror_vars[master_var.op] = []
        original_var_device = tf.DeviceSpec.from_string(
            var_op_name_to_original_device_str[master_var.op.name])
        mirror_var_device = tf.DeviceSpec(job=worker_device.job,
                                          replica=worker_device.replica,
                                          task=worker_device.task)
        if original_var_device.device_type in ['CPU', 'cpu']:
            # place replicated variable on CPU
            mirror_var_device.device_type = 'CPU'
            mirror_var_device.device_index = 0
            with tf.device(mirror_var_device):
                mirror_var = tf.get_variable(
                    ops.prepend_name_scope(master_var.op.name,
                                           parallax_replica_prefix('cpu')),
                    dtype=master_var.dtype.base_dtype,
                    initializer=master_var.initial_value,
                    trainable=False,
                    collections=[tf.GraphKeys.LOCAL_VARIABLES])
                master_var_op_to_mirror_vars[master_var.op].append(mirror_var)
                mirror_variable_init_ops.append(
                    tf.assign(mirror_var, master_var))
        else:
            for i in range(num_replicas_per_worker):
                # place replicated variable on each device(GPU)
                mirror_var_device.device_type = 'GPU'
                mirror_var_device.device_index = i
                with tf.device(mirror_var_device):
                    mirror_var = tf.get_variable(
                        ops.prepend_name_scope(master_var.op.name,
                                               parallax_replica_prefix(i)),
                        dtype=master_var.dtype.base_dtype,
                        initializer=master_var.initial_value,
                        trainable=False,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES])
                master_var_op_to_mirror_vars[master_var.op].append(mirror_var)
                mirror_variable_init_ops.append(
                    tf.assign(mirror_var, master_var))

        # Make all consumers of the value of master variable
        # use the mirror variable.
        # Get a copy of consumers list to avoid incorrect list iteration.
        master_var_consumers = [c for c in master_var.value().consumers()]
        for consumer_op in master_var_consumers:
            if consumer_op in mirror_variable_init_ops:
                continue
            elif consumer_op.name.startswith(PARALLAX_REPLICA_PREFIX):
                device_index = 0
                if len(master_var_op_to_mirror_vars[master_var.op]) > 1:
                    # Mirror variables are created on GPU,
                    # find one on the same GPU.
                    device_index = int(consumer_op.name.split(
                        PARALLAX_REPLICA_PREFIX)[1].split('/')[0])
                    update_consumers(
                        [consumer_op],
                        old_tensor=master_var.value(),
                        new_tensor=master_var_op_to_mirror_vars[master_var.op][device_index].value())
            else:
                parallax_log.debug(
                    "Consumer %s of value of variable %s is a shared node, "
                    "do not change to mirror variable"
                    % (consumer_op.name, master_var.op.name))

    # Update operations colocated with master variables to be colocated
    # with mirror variables
    for op in tf.get_default_graph().get_operations():
        # Do not update shared node
        if not op.name.startswith(PARALLAX_REPLICA_PREFIX):
            continue
        new_colocation_group = []
        for colocation_group in op.colocation_groups():
            assert colocation_group.startswith(\
                BINARY_ENCODED_COLOCATION_PREFIX)
            current_binding_op_name = \
                colocation_group[len(BINARY_ENCODED_COLOCATION_PREFIX):]\
                .decode("ascii")
            current_binding_op = tf.get_default_graph()\
                .get_operation_by_name(current_binding_op_name)
            if current_binding_op in master_var_op_to_mirror_vars:
                device_index = 0
                if len(master_var_op_to_mirror_vars[current_binding_op]) > 1:
                    # Mirror variables are created on GPU, find one one the
                    # same GPU
                    device_index = int(op.name
                                       .split(PARALLAX_REPLICA_PREFIX)[1]
                                       .split('/')[0])
                op_name_to_bind_to = \
                     COLOCATION_PREFIX \
                     + master_var_op_to_mirror_vars[current_binding_op][device_index].op.name
                new_colocation_group.append(compat.as_bytes(op_name_to_bind_to))
            else:
                new_colocation_group.append(colocation_group)
        op._set_attr("_class", attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(s=new_colocation_group)))

    with tf.control_dependencies(mirror_variable_init_ops):
        tf.no_op(name=MIRROR_VARIABLE_INIT_OP)
    return master_var_op_to_mirror_vars


def update_shard_values_for_worker(num_workers, worker_id):
    num_shards_per_worker = 1
    for num_shards in tf.get_collection(shard.NUM_SHARDS):
        num_shards_tensor = num_shards.op.node_def.attr["value"].tensor
        num_shards_per_worker = \
            num_shards_tensor.int64_val[0]
        num_shards_tensor.int64_val[0] *= num_workers
        num_shards.op._set_attr("value", attr_value_pb2.AttrValue(tensor=num_shards_tensor))
        assert num_shards.op.node_def.attr["value"].tensor.int64_val[0] == num_shards_per_worker * num_workers

    for shard_id in tf.get_collection(shard.SHARD_ID):
        shard_id_tensor = shard_id.op.node_def.attr["value"].tensor
        shard_id_tensor.int64_val[0] += \
            num_shards_per_worker * worker_id
        shard_id.op._set_attr("value", attr_value_pb2.AttrValue(tensor=shard_id_tensor))
        assert shard_id.op.node_def.attr["value"].tensor.int64_val[0] == shard_id_tensor.int64_val[0]

    # find and update dataset with shard filter predicate
    if len(tf.get_collection(shard.SHARD_FILTER_PRED)) > 0:
        shard_filter_pred_names = [v.decode("ascii") \
                for v in tf.get_collection(shard.SHARD_FILTER_PRED)]
        for op in tf.get_default_graph().get_operations():
            if 'dataset_factory' not in op.node_def.attr:
                continue
            func_name = op.node_def.attr['dataset_factory'].func.name
            dataset_factory_func = tf.get_default_graph()._get_function(func_name)
            dataset_factory_func_def = dataset_factory_func.definition
            node_name_to_node = {}
            for node in dataset_factory_func_def.node_def:
                node_name_to_node[node.name] = node
                if 'predicate' in node.attr \
                        and (node.attr['predicate'].func.name
                             in shard_filter_pred_names):
                    num_shards_node_name = \
                        node.input[shard.FILTER_DATASET_NUM_SHARDS_POS]\
                            .split(':output:0')[0]
                    shard_id_node_name = \
                        node.input[shard.FILTER_DATASET_SHARD_ID_POS]\
                            .split(':output:0')[0]
                    num_shards_node = node_name_to_node[num_shards_node_name]
                    shard_id_node = node_name_to_node[shard_id_node_name]
                    num_shards_per_worker = \
                        num_shards_node.attr['value'].tensor.int64_val[0]
                    num_shards_node.attr['value'].tensor.int64_val[0] *= \
                        num_workers
                    shard_id_node.attr['value'].tensor.int64_val[0] += \
                        num_shards_per_worker * worker_id
                    if dataset_factory_func._c_func:
                        # update dataset factory name
                        func_name = '%s_%d' % (func_name, shard_id_node.attr['value'].tensor.int64_val[0])
                        dataset_factory_func._func_name = func_name
                        dataset_factory_func_def.signature.name = func_name

                        serialized = dataset_factory_func_def.SerializeToString()
                        c_func = c_api.TF_FunctionImportFunctionDef(serialized)
                        dataset_factory_func._c_func = \
                            c_api_util.ScopedTFFunction(c_func)

                        #TODO: remove old dataset factory function
                        tf.get_default_graph()._add_function(dataset_factory_func)
                        op_func = op.node_def.attr['dataset_factory'].func
                        op_func.name = func_name
                        op._set_attr('dataset_factory', 
                                     attr_value_pb2.AttrValue(func=op_func))
                        break

            assert dataset_factory_func == tf.get_default_graph()._get_function(func_name)

def _get_shared_name_to_stage_ops(ops):
    stage_ops = [op for op in ops if op.type in STAGE_OP_TYPES]
    shared_name_to_stage_ops = {}
    for stage_op in stage_ops:
        shared_name = stage_op.get_attr("shared_name")
        if shared_name not in shared_name_to_stage_ops:
            shared_name_to_stage_ops[shared_name] = []
        shared_name_to_stage_ops[shared_name].append(stage_op)
    return shared_name_to_stage_ops


def get_pipeline_ops(ops):
    unstage_ops = [op for op in ops
                   if op.type in UNSTAGE_OP_TYPES]
    dequeue_ops = [op for op in ops
                   if op.type in DEQUEUE_OP_TYPES]
    iterator_ops = [op for op in ops
                    if op.type in ITERATOR_OP_TYPES]

    pipeline_ops = set()
    unstage_dequeue_iterator_queue = []
    stage_enqueue_iterator_ops_queue = []
    unstage_dequeue_iterator_queue.extend(unstage_ops)
    unstage_dequeue_iterator_queue.extend(dequeue_ops)
    unstage_dequeue_iterator_queue.extend(iterator_ops)
    visited = set()
    queue_name_to_queue_runner = {}

    shared_name_to_stage_ops = _get_shared_name_to_stage_ops(
        tf.get_default_graph().get_operations())

    for queue_runner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        queue_name_to_queue_runner[queue_runner.name] = queue_runner

    while (len(unstage_dequeue_iterator_queue) > 0 or
           len(stage_enqueue_iterator_ops_queue) > 0):
        if len(unstage_dequeue_iterator_queue) > 0:
            curr_op = unstage_dequeue_iterator_queue.pop()
            if curr_op in visited:
                continue
            visited.add(curr_op)

            if curr_op.type in UNSTAGE_OP_TYPES:
                stage_shared_name = curr_op.get_attr("shared_name")
                stage_ops = shared_name_to_stage_ops[stage_shared_name]
                for stage_op in stage_ops:
                    pipeline_ops.add(stage_op)
                    stage_enqueue_iterator_ops_queue.append(stage_op)
                # Handle colocation groups of unstage op (NoOp)
                assert len(curr_op.colocation_groups()) == 1
                stage_no_op_name = curr_op.colocation_groups()[0][
                                   len(BINARY_ENCODED_COLOCATION_PREFIX):].decode("ascii")
                pipeline_ops.add(tf.get_default_graph()
                                 .get_operation_by_name(stage_no_op_name))
            elif curr_op.type in DEQUEUE_OP_TYPES:
                queue_ops = [input.op for input in curr_op.inputs
                             if input.op.type in QUEUE_OP_TYPES]
                assert len(queue_ops) == 1
                queue_op = queue_ops[0]
                queue_runner = queue_name_to_queue_runner[queue_op.name]
                for enqueue_op in queue_runner.enqueue_ops:
                    pipeline_ops.add(enqueue_op)
                    stage_enqueue_iterator_ops_queue.append(enqueue_op)
                pipeline_ops.add(queue_runner.close_op)
                pipeline_ops.add(queue_runner.cancel_op)
            elif curr_op.type in ITERATOR_OP_TYPES:
                consumer_ops = get_consumers(curr_op)
                stage_enqueue_iterator_ops_queue.extend(consumer_ops)
            else:
                raise RuntimeError("Should not reach here")

        elif len(stage_enqueue_iterator_ops_queue) > 0:
            curr_op = stage_enqueue_iterator_ops_queue.pop()
            if curr_op in visited:
                continue
            visited.add(curr_op)
            ancestor_ops = get_ancestors([curr_op],
                                          include_control_inputs=True)
            for ancestor_op in ancestor_ops:
                pipeline_ops.add(ancestor_op)
                if ancestor_op.type in \
                    UNSTAGE_OP_TYPES + DEQUEUE_OP_TYPES + ITERATOR_OP_TYPES:
                    unstage_dequeue_iterator_queue.append(ancestor_op)

    return pipeline_ops


def construct_multi_gpu_graph_def(single_gpu_graph_def,
                                  op_names_to_replicate,
                                  op_names_to_share,
                                  num_replicas,
                                  tensor_or_op_name_to_replica_names):
    def _update_colocation(node, replica_id=None):
        if '_class' not in node.attr:
            return
        class_list = node.attr['_class'].list
        to_delete = []
        for i in range(len(class_list.s)):
            s = class_list.s[i]
            if s.startswith(BINARY_ENCODED_COLOCATION_PREFIX):
                op_name_to_bind_to = s[len(BINARY_ENCODED_COLOCATION_PREFIX):].decode("ascii")
                if op_name_to_bind_to in op_names_to_replicate:
                    # delete colocation constraint if shared op needs to be
                    # colocated with replica op
                    if replica_id is None:
                        to_delete.append(s)
                    else:
                        new_op_name_to_bind_to = \
                            ops.prepend_name_scope(
                                op_name_to_bind_to,
                                parallax_replica_prefix(replica_id))
                        class_list.s[i] = compat.as_bytes(
                            '%s%s' % (COLOCATION_PREFIX,
                                      new_op_name_to_bind_to))
        for item in to_delete:
            class_list.s.remove(item)

    multi_gpu_graph_def = graph_pb2.GraphDef()
    multi_gpu_graph_def.library.Clear()
    multi_gpu_graph_def.library.CopyFrom(single_gpu_graph_def.library)
    for node in single_gpu_graph_def.node:
        if node.name in op_names_to_share:
            multi_gpu_graph_def.node.extend([node])  # copy
            new_node = multi_gpu_graph_def.node[-1]
            for i in range(len(new_node.input)):
                if _get_op_name(new_node.input[i]) in op_names_to_replicate:
                    new_node.input[i] = \
                        ops.prepend_name_scope(new_node.input[i],
                                               parallax_replica_prefix(0))
            _update_colocation(new_node)
            tensor_or_op_name_to_replica_names.extend_mapping_from_nodedef(node, node)
        elif node.name in op_names_to_replicate:
            for replica_id in range(num_replicas):
                multi_gpu_graph_def.node.extend([node])  # copy
                new_node = multi_gpu_graph_def.node[-1]
                new_node.name = \
                    ops.prepend_name_scope(new_node.name,
                                           parallax_replica_prefix(replica_id))
                if 'CPU' not in new_node.device \
                        and 'cpu' not in new_node.device:
                    new_node.device = \
                        '%s/device:GPU:%d' \
                        % (new_node.device[:new_node.device.find('/device')],
                           replica_id)
                for i in range(len(new_node.input)):
                    if _get_op_name(new_node.input[i]) in op_names_to_replicate:
                        new_node.input[i] = \
                            ops.prepend_name_scope(
                                new_node.input[i],
                                parallax_replica_prefix(replica_id))
                if new_node.op in STAGE_OP_TYPES + UNSTAGE_OP_TYPES:
                    new_node.attr['shared_name'].s = compat.as_bytes(
                        ops.prepend_name_scope(
                            compat.as_str(new_node.attr['shared_name'].s),
                            parallax_replica_prefix(replica_id)))
                _update_colocation(new_node, replica_id)
                if 'frame_name' in new_node.attr:
                    new_node.attr['frame_name'].s = compat.as_bytes(
                        ops.prepend_name_scope(
                            compat.as_str(new_node.attr['frame_name'].s),
                            parallax_replica_prefix(replica_id)))
                tensor_or_op_name_to_replica_names.extend_mapping_from_nodedef(node, new_node)
        else:
            raise RuntimeError("Should not reach here")

    return multi_gpu_graph_def


def get_new_col_def_of_node_list(col_def, op_names_to_replicate, num_replicas):
    new_col_def = meta_graph_pb2.CollectionDef()
    for tensor_name in col_def.node_list.value:
        if _get_op_name(tensor_name) in op_names_to_replicate:
            new_col_def.node_list.value.extend(
                [ops.prepend_name_scope(tensor_name, parallax_replica_prefix(i))
                 for i in range(num_replicas)])
        else:
            new_col_def.node_list.value.append(tensor_name)
    return new_col_def


def update_queue_runners(multi_gpu_meta_graph_def, op_names_to_replicate,
                          num_replicas):
    def _get_new_qr_def(qr_def, prefix, only_rename_enqueue_ops):
        new_qr_def = queue_runner_pb2.QueueRunnerDef()
        new_qr_def.CopyFrom(qr_def)
        del new_qr_def.enqueue_op_name[:]
        for enqueue_op_name in qr_def.enqueue_op_name:
            new_qr_def.enqueue_op_name.append(
                ops.prepend_name_scope(enqueue_op_name, prefix))
        if not only_rename_enqueue_ops:
            new_qr_def.queue_name = \
                ops.prepend_name_scope(qr_def.queue_name, prefix)
            new_qr_def.close_op_name = \
                ops.prepend_name_scope(qr_def.close_op_name, prefix)
            new_qr_def.cancel_op_name = \
                ops.prepend_name_scope(qr_def.cancel_op_name, prefix)
        return new_qr_def

    if tf.GraphKeys.QUEUE_RUNNERS not in multi_gpu_meta_graph_def.collection_def:
        return

    qr_collection = \
        multi_gpu_meta_graph_def.collection_def[tf.GraphKeys.QUEUE_RUNNERS]
    new_qr_col = meta_graph_pb2.CollectionDef()
    for qr_def_string in qr_collection.bytes_list.value:
        qr_def = queue_runner_pb2.QueueRunnerDef()
        qr_def.ParseFromString(qr_def_string)
        assert len(qr_def.enqueue_op_name) > 0
        if qr_def.enqueue_op_name[0] in op_names_to_replicate:
            if qr_def.queue_name in op_names_to_replicate:
                new_qr_defs = \
                    [_get_new_qr_def(qr_def, parallax_replica_prefix(i), False)
                     for i in range(num_replicas)]
            else:
                new_qr_defs = \
                    [_get_new_qr_def(qr_def, parallax_replica_prefix(i), True)
                     for i in range(num_replicas)]
            new_qr_col.bytes_list.value.extend([new_qr_def.SerializeToString()
                                                for new_qr_def in new_qr_defs])
        else:
            new_qr_col.bytes_list.value.append(qr_def.SerializeToString())
    multi_gpu_meta_graph_def.collection_def[tf.GraphKeys.QUEUE_RUNNERS].Clear()
    multi_gpu_meta_graph_def.collection_def[tf.GraphKeys.QUEUE_RUNNERS]\
        .CopyFrom(new_qr_col)


def update_local_variables(multi_gpu_meta_graph_def, op_names_to_replicate,
                            num_replicas):
    def _get_new_var_def(var_def, prefix):
        new_var_def = variable_pb2.VariableDef()
        new_var_def.CopyFrom(var_def)
        new_var_def.variable_name = \
            ops.prepend_name_scope(var_def.variable_name, prefix)
        new_var_def.initializer_name = \
            ops.prepend_name_scope(var_def.initializer_name, prefix)
        new_var_def.snapshot_name = \
            ops.prepend_name_scope(var_def.snapshot_name, prefix)
        return new_var_def

    if tf.GraphKeys.LOCAL_VARIABLES not in multi_gpu_meta_graph_def.collection_def:
        return

    lv_collection = \
        multi_gpu_meta_graph_def.collection_def[tf.GraphKeys.LOCAL_VARIABLES]
    new_lv_col = meta_graph_pb2.CollectionDef()
    for var_def_string in lv_collection.bytes_list.value:
        var_def = variable_pb2.VariableDef()
        var_def.ParseFromString(var_def_string)
        if _get_op_name(var_def.variable_name) in op_names_to_replicate:
            new_var_defs = \
                [_get_new_var_def(var_def, parallax_replica_prefix(i))
                 for i in range(num_replicas)]
            new_lv_col.bytes_list.value.extend(
                [new_var_def.SerializeToString()
                 for new_var_def in new_var_defs])
        else:
            new_lv_col.bytes_list.value.append(var_def.SerializeToString())
    multi_gpu_meta_graph_def.collection_def[tf.GraphKeys.LOCAL_VARIABLES]\
        .Clear()
    multi_gpu_meta_graph_def.collection_def[tf.GraphKeys.LOCAL_VARIABLES]\
        .CopyFrom(new_lv_col)
    if len(lv_collection.bytes_list.value) == 0:
        del multi_gpu_meta_graph_def\
            .collection_def[tf.GraphKeys.LOCAL_VARIABLES]


def add_aggregate_gradients_ops(multi_gpu_meta_graph_def,
                                op_names_to_replicate,
                                op_to_control_consumer_ops,
                                gradient_info_list,
                                num_replicas,
                                average_option):
    def _get_aggregated_dense_grad(grad_name):
        grad_op_name = _get_op_name(grad_name)
        output_idx = int(grad_name.split(':')[1])
        assert grad_op_name in op_names_to_replicate
        grad_ops = [tf.get_default_graph().get_operation_by_name(
                    ops.prepend_name_scope(grad_op_name,
                                           parallax_replica_prefix(i)))
                    for i in range(num_replicas)]
        # Aggregate gradients on CPU
        with tf.device('/device:CPU:0'):
            grad_sum_op_name = \
                ops.prepend_name_scope(grad_op_name, u"%sAdd" % PARALLAX_PREFIX)
            grad_sum = \
                tf.add_n([grad_op.outputs[output_idx] for grad_op in grad_ops],
                         name=grad_sum_op_name)
            grad_avg_op_name =\
                ops.prepend_name_scope(grad_op_name, u"%sDiv" % PARALLAX_PREFIX)
            grad_avg = tf.realdiv(grad_sum, num_replicas, name=grad_avg_op_name)
        return grad_avg

    def _get_aggregated_sparse_grad(var_op, indices_name, values_name,
                                    dense_shape_name):
        indices_op_name = _get_op_name(indices_name)
        values_op_name = _get_op_name(values_name)
        dense_shape_op_name = _get_op_name(dense_shape_name)
        indices_output_idx = int(indices_name.split(':')[1])
        values_output_idx = int(values_name.split(':')[1])
        dense_shape_output_idx = int(dense_shape_name.split(':')[1])
        assert indices_op_name in op_names_to_replicate
        assert values_op_name in op_names_to_replicate
        assert dense_shape_op_name in op_names_to_replicate
        indices_ops = [tf.get_default_graph().get_operation_by_name(
            ops.prepend_name_scope(indices_op_name, parallax_replica_prefix(i)))
            for i in range(num_replicas)]
        values_ops = [tf.get_default_graph().get_operation_by_name(
            ops.prepend_name_scope(values_op_name, parallax_replica_prefix(i)))
            for i in range(num_replicas)]
        dense_shape_ops = [tf.get_default_graph().get_operation_by_name(
            ops.prepend_name_scope(dense_shape_op_name,
                                   parallax_replica_prefix(i)))
            for i in range(num_replicas)]
        indexed_slices_grads = [tf.IndexedSlices(
            values_op.outputs[values_output_idx],
            indices_op.outputs[indices_output_idx],
            dense_shape_op.outputs[dense_shape_output_idx])
            for indices_op, values_op, dense_shape_op
            in zip(indices_ops, values_ops, dense_shape_ops)]
        # Aggregate gradients on CPU
        with tf.device('/device:CPU:0'):
            grad_accum_op_name = \
                ops.prepend_name_scope(values_op_name,
                                       u"%sAccum" % PARALLAX_PREFIX)
            grad_accum = tf.SparseConditionalAccumulator(
                dtype=indexed_slices_grads[0].values.dtype,
                shape=var_op.outputs[0].shape,
                shared_name=grad_accum_op_name,
                name=grad_accum_op_name)
            accum_apply_ops = [grad_accum.apply_indexed_slices_grad(
                indexed_slices_grads[i],
                MAX_INT64,
                name=ops.prepend_name_scope(
                        values_op_name,
                        u"%s-Accum-Apply" % parallax_replica_prefix(i)))
                        for i in range(num_replicas)]
            take_grad_op_name = ops.prepend_name_scope(
                values_op_name,
                u"%sTake-Grad" % PARALLAX_PREFIX)
            with tf.control_dependencies(accum_apply_ops):
                take_grad = grad_accum.take_indexed_slices_grad(
                    num_replicas,
                    average_option=average_option,
                    name=take_grad_op_name)
            new_indices = take_grad.indices
            new_values = take_grad.values
            new_dense_shape = take_grad.dense_shape
            if indexed_slices_grads[0].indices.dtype != new_indices.dtype:
                new_indices = tf.cast(
                    new_indices,
                    indexed_slices_grads[0].indices.dtype,
                    name=ops.prepend_name_scope(
                        values_op_name,
                        u"%sTake-Grad-Cast-Indices" % PARALLAX_PREFIX)
                )
            if indexed_slices_grads[0].dense_shape.dtype \
                    != new_dense_shape.dtype:
                new_dense_shape = tf.cast(
                    new_dense_shape,
                    indexed_slices_grads[0].dense_shape.dtype,
                    name=ops.prepend_name_scope(
                        values_op_name,
                        u"%sTake-Grad-Cast-Shape" % PARALLAX_PREFIX)
                )
        return tf.IndexedSlices(new_values, new_indices, new_dense_shape)

    def _update_gradient_consumers(consumer_op_names, control_consumer_op_names,
                                   old_tensor_name, new_tensor):
        graph = tf.get_default_graph()
        consumer_ops = [graph.get_operation_by_name(name)
                        for name in consumer_op_names]
        control_consumer_ops = [graph.get_operation_by_name(name)
                                for name in control_consumer_op_names]
        # Gradient consumers are currently using gradient generated
        # from replica 0
        old_op_name = _get_op_name(old_tensor_name)
        replica_0_op_name = ops.prepend_name_scope(
            old_op_name,
            parallax_replica_prefix(0))
        output_idx = int(old_tensor_name.split(':')[1])
        replica_0_op = graph.get_operation_by_name(replica_0_op_name)
        replica_0_tensor = replica_0_op.outputs[output_idx]
        update_consumers(consumer_ops, replica_0_tensor, new_tensor)
        update_control_consumers(control_consumer_ops, replica_0_tensor.op,
                                  new_tensor.op)

    with tf.Graph().as_default() as graph:
        import_start_time = time.time()
        tf.train.import_meta_graph(multi_gpu_meta_graph_def)
        import_duration = time.time() - import_start_time
        parallax_log.debug(
            "Time to import multi-GPU meta graph : %.3f seconds"
            % import_duration)
        trainable_variable_ops = \
            [var.op.name for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        # Aggregate grads
        for gradient_info in gradient_info_list:
            var_op = gradient_info._target.op
            assert var_op.name in trainable_variable_ops
            grad = gradient_info._grad
            if isinstance(grad, tf.Tensor):
                values_name = grad.name
                agg_grad = _get_aggregated_dense_grad(values_name)
                cc_names = [c.name for c in op_to_control_consumer_ops[grad.op]]
                c_names = [c.name for c in grad.consumers()]
                _update_gradient_consumers(
                    c_names,
                    cc_names,
                    values_name,
                    agg_grad)
            elif isinstance(grad, tf.IndexedSlices):
                indices_name, values_name, dense_shape_name = \
                    (grad.indices.name, grad.values.name, grad.dense_shape.name)
                agg_grad = _get_aggregated_sparse_grad(
                    var_op, indices_name, values_name, dense_shape_name)

                indices_cc_names = [c.name for c in op_to_control_consumer_ops[grad.indices.op]]
                values_cc_names = [c.name for c in op_to_control_consumer_ops[grad.values.op]]
                indices_c_names = [c.name for c in grad.indices.consumers()]
                values_c_names = [c.name for c in grad.values.consumers()]
                _update_gradient_consumers(
                    indices_c_names,
                    indices_cc_names,
                    indices_name,
                    agg_grad.indices)
                _update_gradient_consumers(
                    values_c_names,
                    list(set(values_cc_names).difference(indices_cc_names)),
                    values_name,
                    agg_grad.values)
            else:
                raise RuntimeError("Incorrect grad.")

            gi = GradientsInfo(var_op.outputs[0], agg_grad)
            tf.add_to_collection(tf.GraphKeys.GRADIENTS_INFO, gi)

        return tf.train.export_meta_graph()

def update_shard_info_for_in_graph(meta_graph_def, num_replicas):
    # TODO : cleanup the function code structure
    if num_replicas <= 1:
        return

    node_name_to_node = {}
    for node in meta_graph_def.graph_def.node:
        node_name_to_node[node.name] = node

    if shard.SHARD_ID in meta_graph_def.collection_def:
        shard_id_node_names = \
            meta_graph_def.collection_def[shard.SHARD_ID].node_list.value
        num_shard_id_nodes = len(shard_id_node_names)
        if num_shard_id_nodes == num_replicas:
            for shard_id_node_name in shard_id_node_names:
                parallax_log.debug(shard_id_node_name)
                shard_id_to_update = \
                    int(shard_id_node_name.split(PARALLAX_REPLICA_PREFIX)[1]
                        .split('/')[0])
                node_name_to_node[_get_op_name(shard_id_node_name)]\
                    .attr['value'].tensor.int64_val[0] = shard_id_to_update
        elif num_shard_id_nodes != 1:
            raise ValueError(
                "The number of shard_id must be same as the number of "
                "replicas or 1")

    if shard.NUM_SHARDS in meta_graph_def.collection_def:
        num_shards_node_names = meta_graph_def.collection_def[shard.NUM_SHARDS]\
                                    .node_list.value
        num_num_shards_nodes = len(num_shards_node_names)
        if num_num_shards_nodes == num_replicas:
            for num_shards_node_name in num_shards_node_names:
                node_name_to_node[_get_op_name(num_shards_node_name)]\
                    .attr['value'].tensor.int64_val[0] = num_replicas
        elif num_num_shards_nodes != 1:
            raise ValueError(
                "The number of num_shards must be same as the number "
                "of replicas or 1")

    # update dataset factory if it uses shard and its consumer node is a replica
    if shard.SHARD_FILTER_PRED in meta_graph_def.collection_def:
        shard_filter_pred_names = \
            [v.decode("ascii") for v in \
            meta_graph_def.collection_def[shard.SHARD_FILTER_PRED].bytes_list.value]
        dataset_factory_replica_consumers = {}
        # collect dataset factory consumers if they are replicas
        for node in meta_graph_def.graph_def.node:
            if 'dataset_factory' in node.attr \
                    and node.name.startswith(PARALLAX_REPLICA_PREFIX):
                dataset_factory_name = node.attr['dataset_factory'].func.name
                if dataset_factory_name \
                        not in dataset_factory_replica_consumers:
                    dataset_factory_replica_consumers[dataset_factory_name] = []
                dataset_factory_replica_consumers[dataset_factory_name]\
                    .append(node)

        updated_lib = function_pb2.FunctionDefLibrary()

        # TODO: Polish this block. Too deeply nested.
        for func in meta_graph_def.graph_def.library.function:
            if func.signature.name in dataset_factory_replica_consumers:
                replicate = False
                for node in func.node_def:
                    if 'predicate' in node.attr \
                            and (node.attr['predicate'].func.name
                                 in shard_filter_pred_names):
                        num_shards_name = \
                            node.input[shard.FILTER_DATASET_NUM_SHARDS_POS]\
                            .split(':output:0')[0]
                        shard_id_name = node.input[shard.FILTER_DATASET_SHARD_ID_POS]\
                            .split(':output:0')[0]

                        for replica_id in range(num_replicas):
                            replica_func = function_pb2.FunctionDef()
                            replica_func.CopyFrom(func)
                            replica_func.signature.name = \
                                ops.prepend_name_scope(
                                    func.signature.name,
                                    parallax_replica_prefix(replica_id))
                            for node in replica_func.node_def:
                                if node.name == num_shards_name:
                                    node.attr['value'].tensor.int64_val[0] = \
                                        num_replicas
                                elif node.name == shard_id_name:
                                    node.attr['value'].tensor.int64_val[0] = \
                                        replica_id
                            updated_lib.function.extend([replica_func])

                        for consumer in dataset_factory_replica_consumers[func.signature.name]:
                            replica_id = int(consumer.name.split(PARALLAX_REPLICA_PREFIX)[1].split('/')[0])
                            replica_func_name = ops.prepend_name_scope(func.signature.name,
                                                                       parallax_replica_prefix(replica_id))
                            consumer.attr['dataset_factory'].func.name = replica_func_name
                        replicate = True
                        break
                if not replicate:
                    updated_lib.function.extend([func])
            else:
                updated_lib.function.extend([func])

        meta_graph_def.graph_def.library.CopyFrom(updated_lib)

def set_boundary_between_workers_and_servers():
  parallax_log.debug('set_boundary_between_workers_and_servers')
  target_op_types = ['Slice', 'Gather', 'L2Loss', 'Size']
  def _move_op(job_to_move, op):
    for inp in op.inputs:
        skip = False
        for colocation_group in inp.op.colocation_groups():
            assert colocation_group.startswith(\
                BINARY_ENCODED_COLOCATION_PREFIX)
            current_binding_op_name = \
                colocation_group[len(BINARY_ENCODED_COLOCATION_PREFIX):]\
                .decode("ascii")
            current_binding_op = tf.get_default_graph()\
                .get_operation_by_name(current_binding_op_name)
            if current_binding_op != inp.op and tf.DeviceSpec.from_string(current_binding_op.device).job == job_to_move:
              skip = True
              break

        if skip:
          continue
        if tf.DeviceSpec.from_string(inp.op.device).job == job_to_move:
          if ((inp.op.type == 'Cast' and inp.op.inputs[0].dtype.size < inp.op.outputs[0].dtype.size)):
              parallax_log.debug('inp op : %s, %s -> %s' % (inp.op.name, inp.op.device, op.device))
              inp.op._set_device(op.device)

    for i in range(len(op.outputs)):
        for consumer in op.outputs[i].consumers():
          skip = False
          for colocation_group in consumer.colocation_groups():
                assert colocation_group.startswith(\
                    BINARY_ENCODED_COLOCATION_PREFIX)
                current_binding_op_name = \
                   colocation_group[len(BINARY_ENCODED_COLOCATION_PREFIX):]\
                   .decode("ascii")
                current_binding_op = tf.get_default_graph()\
                    .get_operation_by_name(current_binding_op_name)
                if current_binding_op != consumer and tf.DeviceSpec.from_string(current_binding_op.device).job == job_to_move:
                  skip = True
                  break

          if skip:
            continue
    
          if tf.DeviceSpec.from_string(consumer.device).job == job_to_move:
              if ((consumer.type == 'Cast' and consumer.inputs[0].dtype.size > consumer.outputs[0].dtype.size) or
                 (consumer.type in target_op_types)):
                 parallax_log.debug('consumer op : %s, %s -> %s' % (consumer.name, consumer.device, op.device))
                 consumer._set_device(op.device)

  for op in tf.get_default_graph().get_operations():
    device = tf.DeviceSpec.from_string(op.device)
    if device.job == 'worker':
      job_to_move = 'ps'
    else:
      job_to_move = 'worker'
    _move_op(job_to_move, op)

def add_aggregate_gradients_ops_only_between(multi_gpu_meta_graph_def,
                                             op_names_to_replicate,
                                             op_to_control_consumer_ops,
                                             gradient_info_list,
                                             num_replicas,
                                             average_option,
                                             local_chief_id,
                                             num_local_worker,
                                             only_sparse,
                                             local_aggregation):
    def _get_aggregated_dense_grad(var_op, grad):
        grad_op_name = grad.op.name
        grad_op = tf.get_default_graph().get_operation_by_name(
            ops.prepend_name_scope(grad_op_name,
                                   parallax_replica_prefix(0)))
        grad = grad_op.outputs[0]
        with tf.device('/job:worker/task:%d/device:CPU:0' % local_chief_id):
            grad_accum = tf.ConditionalAccumulator(
                grad.dtype,
                shape=var_op.outputs[0].get_shape(),
                shared_name=var_op.name + "/grad_accum")
            # Get a copy of consumers list before creating accum_apply_op
            grad_consumers = [c for c in grad.consumers()]
            accum_apply_op = grad_accum.apply_grad(
                grad, local_step=MAX_INT64,
                name=grad.op.name + '_accum_apply_grad')
            with tf.control_dependencies([accum_apply_op]):
                agg_grad = grad_accum.take_grad(num_local_worker,
                                                name=var_op.name + '_take_grad')

        return agg_grad, accum_apply_op

    def _get_aggregated_sparse_grad(var_op, indices_name, values_name,
                                    dense_shape_name):
        indices_op_name = _get_op_name(indices_name)
        values_op_name = _get_op_name(values_name)
        dense_shape_op_name = _get_op_name(dense_shape_name)
        indices_output_idx = int(indices_name.split(':')[1])
        values_output_idx = int(values_name.split(':')[1])
        dense_shape_output_idx = int(dense_shape_name.split(':')[1])
        assert indices_op_name in op_names_to_replicate
        assert values_op_name in op_names_to_replicate
        assert dense_shape_op_name in op_names_to_replicate
        indices_ops = [tf.get_default_graph().get_operation_by_name(
            ops.prepend_name_scope(indices_op_name, parallax_replica_prefix(i)))
            for i in range(num_replicas)]
        values_ops = [tf.get_default_graph().get_operation_by_name(
            ops.prepend_name_scope(values_op_name, parallax_replica_prefix(i)))
            for i in range(num_replicas)]
        dense_shape_ops = [tf.get_default_graph().get_operation_by_name(
            ops.prepend_name_scope(dense_shape_op_name,
                                   parallax_replica_prefix(i)))
            for i in range(num_replicas)]
        indexed_slices_grads = [tf.IndexedSlices(
            values_op.outputs[values_output_idx],
            indices_op.outputs[indices_output_idx],
            dense_shape_op.outputs[dense_shape_output_idx])
            for indices_op, values_op, dense_shape_op
            in zip(indices_ops, values_ops, dense_shape_ops)]
        # Aggregate gradients on CPU
        with tf.device('/job:worker/task:%d/device:CPU:0' % local_chief_id):
            grad_accum_op_name = \
                ops.prepend_name_scope(values_op_name,
                                       u"%sAccum" % PARALLAX_PREFIX)
            grad_accum = tf.SparseConditionalAccumulator(
                dtype=indexed_slices_grads[0].values.dtype,
                shape=var_op.outputs[0].shape,
                shared_name=grad_accum_op_name,
                name=grad_accum_op_name)
            accum_apply_ops = [grad_accum.apply_indexed_slices_grad(
                indexed_slices_grads[i],
                MAX_INT64,
                name=ops.prepend_name_scope(
                        values_op_name,
                        u"%s-Accum-Apply" % parallax_replica_prefix(i)))
                        for i in range(num_replicas)]
            take_grad_op_name = ops.prepend_name_scope(
                values_op_name,
                u"%sTake-Grad" % PARALLAX_PREFIX)
            with tf.control_dependencies(accum_apply_ops):
                take_grad = grad_accum.take_indexed_slices_grad(
                    num_local_worker,
                    average_option=average_option,
                    name=take_grad_op_name)
            new_indices = take_grad.indices
            new_values = take_grad.values
            new_dense_shape = take_grad.dense_shape
            if indexed_slices_grads[0].indices.dtype != new_indices.dtype:
                new_indices = tf.cast(
                    new_indices,
                    indexed_slices_grads[0].indices.dtype,
                    name=ops.prepend_name_scope(
                        values_op_name,
                        u"%sTake-Grad-Cast-Indices" % PARALLAX_PREFIX)
                )
            if indexed_slices_grads[0].dense_shape.dtype \
                    != new_dense_shape.dtype:
                new_dense_shape = tf.cast(
                    new_dense_shape,
                    indexed_slices_grads[0].dense_shape.dtype,
                    name=ops.prepend_name_scope(
                        values_op_name,
                        u"%sTake-Grad-Cast-Shape" % PARALLAX_PREFIX)
                )
        return tf.IndexedSlices(new_values, new_indices, new_dense_shape)

    def _update_gradient_consumers(consumer_op_names, control_consumer_op_names,
                                   old_tensor_name, new_tensor):
        graph = tf.get_default_graph()
        consumer_ops = [graph.get_operation_by_name(name)
                        for name in consumer_op_names]
        control_consumer_ops = [graph.get_operation_by_name(name)
                                for name in control_consumer_op_names]
        # Gradient consumers are currently using gradient generated
        # from replica 0
        old_op_name = _get_op_name(old_tensor_name)
        replica_0_op_name = ops.prepend_name_scope(
            old_op_name,
            parallax_replica_prefix(0))
        output_idx = int(old_tensor_name.split(':')[1])
        replica_0_op = graph.get_operation_by_name(replica_0_op_name)
        replica_0_tensor = replica_0_op.outputs[output_idx]
        update_consumers(consumer_ops, replica_0_tensor, new_tensor)
        update_control_consumers(control_consumer_ops, replica_0_tensor.op,
                                  new_tensor.op)

    with tf.Graph().as_default() as graph:
        tf.train.import_meta_graph(multi_gpu_meta_graph_def)

        # Aggregate grads
        for gradient_info in gradient_info_list:
            var_op = gradient_info._target.op
            grad = gradient_info._grad
            if only_sparse and isinstance(grad, tf.Tensor):
                grad_op_name = grad.op.name
                grad_op = tf.get_default_graph().get_operation_by_name(
                    ops.prepend_name_scope(grad_op_name,
                                           parallax_replica_prefix(0)))
                agg_grad = grad_op.outputs[0]
                #gi = GradientsInfo(gradient_info._target, grad)
                #tf.add_to_collection(tf.GraphKeys.GRADIENTS_INFO, gi)
                #continue
            elif isinstance(grad, tf.Tensor):
                agg_grad, accum_apply_op = _get_aggregated_dense_grad(var_op, grad)
                cc_names = [c.name for c in op_to_control_consumer_ops[grad.op]]
                c_names = [c.name for c in grad.consumers()]
                _update_gradient_consumers(
                    c_names,
                    cc_names,
                    grad.name,
                    agg_grad)
            elif isinstance(grad, tf.IndexedSlices):
                indices_name, values_name, dense_shape_name = \
                    (grad.indices.name, grad.values.name, grad.dense_shape.name)
                if local_aggregation:
                    agg_grad = _get_aggregated_sparse_grad(
                        var_op, indices_name, values_name, dense_shape_name)

                    indices_cc_names = [c.name for c in op_to_control_consumer_ops[grad.indices.op]]
                    values_cc_names = [c.name for c in op_to_control_consumer_ops[grad.values.op]]
                    indices_c_names = [c.name for c in grad.indices.consumers()]
                    values_c_names = [c.name for c in grad.values.consumers()]
                    _update_gradient_consumers(
                        indices_c_names,
                        indices_cc_names,
                        indices_name,
                        agg_grad.indices)
                    _update_gradient_consumers(
                        values_c_names,
                        list(set(values_cc_names).difference(indices_cc_names)),
                        values_name,
                        agg_grad.values)
                else:
                    def _get_tensor_with_prefix(tensor_name):
                        return tf.get_default_graph().get_tensor_by_name(ops.prepend_name_scope(tensor_name, parallax_replica_prefix(0)))
                    agg_grad = tf.IndexedSlices(_get_tensor_with_prefix(values_name),
                                                _get_tensor_with_prefix(indices_name),
                                                _get_tensor_with_prefix(dense_shape_name))
            else:
                raise RuntimeError("Incorrect grad.")

            gi = GradientsInfo(var_op.outputs[0], agg_grad)
            tf.add_to_collection(tf.GraphKeys.GRADIENTS_INFO, gi)

        return tf.train.export_meta_graph()

def add_sync_op_only_between(worker_id,
                             local_worker_id,
                             machine_id,
                             num_local_workers,
                             num_worker_machines,
                             master_var_op_to_mirror_vars,
                             ps_device,
                             worker_device,
                             average_sparse,
                             tensor_or_op_name_to_replica_names,
                             only_sparse,
                             local_aggregation):
    """Adds additional ops needed for synchronous distributed training into
    current graph.
    Main purpose of additional ops are:
    1. Initialization
    2. Synchronization
    3. Gradient aggregation
    Args:
        worker_id: The worker id
        num_workers: Total number of workers to synchronize
        master_var_op_to_mirror_vars: The dictionary of master variable op name
            -> list of replicated variables, could be None
        worker_device : The worker device string
        average_sparse: Whether to average sparse values or not.
    Returns:
        None
    """

    def _get_accum_apply_and_agg_grad(var, grad, indices, dense_shape):
        var_op = var.op
        num_required = num_worker_machines if local_aggregation else num_worker_machines * num_local_workers
        if indices is None:
            assert False # hybrid does not use this function
            grad_accum = tf.ConditionalAccumulator(
                grad.dtype,
                shape=var_op.outputs[0].get_shape(),
                shared_name=var_op.name + "/grad_accum")
            # Get a copy of consumers list before creating accum_apply_op
            grad_consumers = [c for c in grad.consumers()]
            accum_apply_op = grad_accum.apply_grad(
                grad, local_step=MAX_INT64,
                name=grad.op.name + '_accum_apply_grad')
            agg_grad = grad_accum.take_grad(num_worker_machines,
                                            name=var_op.name + '_take_grad')
            with tf.device(var_op.device):
                global_grad = tf.get_variable(
                    grad.op.name +'_global_grad',
                    dtype=grad.dtype,
                    initializer=var.initial_value,
                    trainable=False)
                assign_global_grad_buf = global_grad.assign(agg_grad)
                assign_global_grad_buf = [assign_global_grad_buf.op]
            read_gg = global_grad.read_value()
            global_grad_read_ops = [read_gg.op]
            update_consumers(grad_consumers, grad, read_gg)
            update_control_consumers(op_to_control_consumer_ops[grad.op],
                                      grad.op, global_grad.op)
        else:
            grad_indexed_slices = tf.IndexedSlices(values=grad, indices=indices,
                                                   dense_shape=dense_shape)
            grad_accum = tf.SparseConditionalAccumulator(
                grad.dtype,
                shape=var_op.outputs[0].get_shape(),#grad.shape,
                shared_name=var_op.name + "/grad_accum")
            # Get a copy of consumers list before creating accum_apply_op
            indices_consumers = [c for c in indices.consumers()]
            grad_consumers = [c for c in grad.consumers()]
            accum_apply_op = grad_accum.apply_indexed_slices_grad(
                grad_indexed_slices, local_step=MAX_INT64,
                name=grad.op.name + '_accum_apply_grad')
            average_option = SPARSE_NO_AVERAGE
            if average_sparse:
                average_option = SPARSE_AVERAGE_BY_COUNTER
            agg_grad = grad_accum.take_indexed_slices_grad(
                num_required, average_option=average_option,
                name=var_op.name + '_take_grad')
            agg_indices = agg_grad.indices
            if indices.dtype != agg_grad.indices.dtype:
                agg_indices = tf.cast(agg_grad.indices, indices.dtype)

            with tf.device(var_op.device):
                global_grad_values = tf.get_variable(
                    grad.op.name +'_parallax_global_grad_values',
                    shape=(14,),
                    dtype=agg_grad.values.dtype,
                    initializer=tf.zeros_initializer(),
                    trainable=False,
                    validate_shape=False)
                global_grad_indices = tf.get_variable(
                    grad.op.name + '_parallax_global_grad_indices',
                    shape=(14,),
                    dtype=agg_indices.dtype,
                    initializer=tf.zeros_initializer(),
                    trainable=False,
                    validate_shape=False)
                global_grad_dense_shape = tf.get_variable(
                    grad.op.name + '_parallax_global_grad_dense_shape',
                    shape=(14,),
                    dtype=agg_grad.dense_shape.dtype,
                    initializer=tf.zeros_initializer(),
                    trainable=False,
                    validate_shape=False)

                assign_global_grad_values = tf.assign(global_grad_values, agg_grad.values, validate_shape=False)
                assign_global_grad_indices = tf.assign(global_grad_indices, agg_indices, validate_shape=False)
                assign_global_grad_dense_shape = tf.assign(global_grad_dense_shape, agg_grad.dense_shape, validate_shape=False)
                assign_global_grad_buf = [assign_global_grad_values.op, assign_global_grad_indices.op, assign_global_grad_dense_shape.op]
                for var in [global_grad_values, global_grad_indices, global_grad_dense_shape]:
                    tf.add_to_collection(PARALLAX_GLOBAL_GRADS, var)
 
                grad_update_sync_queues = \
                        [tf.FIFOQueue(1, [tf.bool], shapes=[[]],
                              name='auto_parallel_%s_update_sync_queue_%d'
                                   % (var_op.name, i),
                              shared_name='auto_parallel_%s'
                                          '_update_sync_queue_%d'
                                          % (global_grad_values.name, i))
                        for i in range(num_worker_machines * num_local_workers)]
                token = tf.constant(False)
                queue_ops = []
                if worker_id == 0:
                  with tf.control_dependencies(assign_global_grad_buf):
                    for i,q in enumerate(grad_update_sync_queues):
                      if i != worker_id:
                          def true_fn():
                              with tf.control_dependencies([q.dequeue()]):
                                  return q.enqueue(token)
                          queue_ops.append(tf.cond(q.size() > 0, true_fn, lambda: q.enqueue(token)))
                      else:
                          queue_ops.append(tf.no_op())
                else:
                    queue_ops.append(grad_update_sync_queues[worker_id].dequeue())
                  
            agg_grad = tf.IndexedSlices(values=global_grad_values,
                                        indices=global_grad_indices,
                                        dense_shape=global_grad_dense_shape)

            assert isinstance(agg_grad, tf.IndexedSlices)
            with tf.control_dependencies(queue_ops):
                read_gg_indices = global_grad_indices.read_value()
                read_gg_values = global_grad_values.read_value()
            global_grad_read_ops = [read_gg_indices.op, read_gg_values.op]
            update_consumers(indices_consumers, indices, read_gg_indices)
            update_consumers(grad_consumers, grad, read_gg_values)
            update_control_consumers(op_to_control_consumer_ops[indices.op],
                                      indices.op, global_grad_indices.op)
            update_control_consumers(op_to_control_consumer_ops[grad.op],
                                      grad.op, global_grad_values.op)
        return accum_apply_op, tf.IndexedSlices(read_gg_indices, read_gg_values), assign_global_grad_buf, \
               queue_ops

    def _get_mirror_variable_update_ops(master_var_op_to_mirror_vars,
                                        grad_apply_finished, var):
        with tf.device(this_worker_cpu):
            with tf.control_dependencies(grad_apply_finished):
                updated_value = var.read_value()
        update_ops = []
        for mirror_var in master_var_op_to_mirror_vars[var.op]:
            with tf.device(mirror_var.device):
                update_ops.append(mirror_var.assign(updated_value))
        return update_ops

    def _replace_update_op_with_read_op(var_op, var_update_op, finish_op):

        var_update_consumers = [c for c in var_update_op.outputs[0].consumers()]
        for consumer in var_update_consumers:
            parallax_log.debug(
                'var: %s, var_update : %s, consumer : %s'
                % (var_op.name, var_update_op.name, consumer.name))
            assert consumer.type not in all_var_update_op_types

        # TODO: exploit locality: read updated value from mirror
        with tf.control_dependencies([finish_op]):
            with tf.device(var_op.device):
                updated_var_value = global_var_op_to_var[var_op].read_value()

        update_consumers(var_update_consumers, var_update_op.outputs[0],
                          updated_var_value)
        tensor_or_op_name_to_replica_names.update_mapping_from_tensor(
            var_update_op.outputs[0], updated_var_value)

    this_worker_cpu = tf.DeviceSpec.from_string(worker_device)
    this_worker_cpu.device_type = 'CPU'
    this_worker_cpu.device_index = 0
    is_chief = worker_id == 0
    is_local_chief = local_worker_id == 0

    trainable_var_op_to_var = \
        dict([(var.op, var)
              for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
    global_var_op_to_var = \
        dict([(var.op, var)
              for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
    op_to_control_consumer_ops = \
        get_all_control_consumers(tf.get_default_graph())

    var_op_to_agg_grad = {}
    var_op_to_accum_apply_op = {}
    var_op_to_sync_deps = {}
    var_op_to_global_grad_read_ops = {}
    sparse_var_ops = []
    # Aggregate gradients from different workers using ConditionalAccumulator.
    # var_op_to_agg_grad and var_op_to_accum_apply_op are updated.
    for gradients_info in tf.get_collection(tf.GraphKeys.GRADIENTS_INFO):
        # grad_tensor == local aggregated gradients
        grad_tensor = gradients_info._grad
        target_tensor = gradients_info._target
        if isinstance(grad_tensor, tf.Tensor) and only_sparse:
            continue

        if target_tensor.op not in trainable_var_op_to_var:
            parallax_log.debug(
                "Gradient for non-trainable variable %s is created, "
                "do not insert accumulator for aggregating this gradient"
                % target_tensor.op.name)
            continue
        var_op = target_tensor.op
        #parallax_log.info(var_op.name)
        if isinstance(grad_tensor, tf.Tensor):
            grad = grad_tensor
            indices = None
            dense_shape = None
        else:
            grad = grad_tensor.values
            indices = grad_tensor.indices
            dense_shape = grad_tensor.dense_shape
            sparse_var_ops.append(var_op)
        with tf.device(var_op.device), tf.name_scope(""):
            accum_apply_op, agg_grad, assign_global_grad_buf,\
             global_grad_sync_ops = \
                _get_accum_apply_and_agg_grad(target_tensor, grad, indices,
                                              dense_shape)
        gradients_info._grad = agg_grad
        if indices == None:
            var_op_to_agg_grad[var_op] = (None, agg_grad)
        else:
            var_op_to_agg_grad[var_op] = (agg_grad.indices, agg_grad.values)

        var_op_to_accum_apply_op[var_op] = accum_apply_op
        if is_local_chief:
            var_op_to_sync_deps[var_op] = [accum_apply_op] + global_grad_sync_ops
        else:
            if not local_aggregation:
                var_op_to_sync_deps[var_op] = grad_tensor.op.control_inputs +\
                    reduce(lambda s, x: s + x.op.control_inputs,
                           grad_tensor.op.inputs, []) + global_grad_sync_ops + [accum_apply_op]
            else:
                var_op_to_sync_deps[var_op] = grad_tensor.op.control_inputs +\
                    reduce(lambda s, x: s + x.op.control_inputs,
                           grad_tensor.op.inputs, []) + global_grad_sync_ops
        if is_chief:
            var_op_to_sync_deps[var_op].extend(assign_global_grad_buf)
    global_step_op = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0].op
    assert len(tf.get_collection(tf.GraphKeys.GLOBAL_STEP)) == 1

    var_op_to_finish_op = {}
    trainable_var_op_to_update_op = {}
    non_trainable_var_op_to_update_op = {}
    all_var_update_op_types = list(sparse_var_update_op_types.keys()) \
        + list(dense_var_update_op_types.keys())

    for op in tf.get_default_graph().get_operations():
        # Find variable update ops
        if not op.type in all_var_update_op_types:
            continue

        var_update_op = op
        var_op = var_update_op.inputs[UPDATE_OP_VAR_POS].op
        if var_op not in global_var_op_to_var \
                or var_update_op == global_var_op_to_var[var_op].initializer:
            continue

        assert var_op not in trainable_var_op_to_update_op
        assert var_op not in non_trainable_var_op_to_update_op

        if var_op in trainable_var_op_to_var:
            trainable_var_op_to_update_op[var_op] = var_update_op
            is_trainable = True
        else:
            non_trainable_var_op_to_update_op[var_op] = var_update_op
            is_trainable = False

        # Even if only_sparse, update ops for dense vars are inserted to
        # trainable_var_op_to_update_op or non_trainable_var_op_to_update_op.
        if only_sparse and var_op not in sparse_var_ops:
            continue

        queue_ops = []
        with tf.device(var_op.device), tf.name_scope(""):
            var_update_global_sync_queues = \
                [tf.FIFOQueue(1, [tf.bool], shapes=[[]],
                              name='auto_parallel_%s_update_sync_queue_%d'
                                   % (var_op.name, i),
                              shared_name='auto_parallel_%s'
                                          '_update_sync_queue_%d'
                                          % (var_op.name, i))
                 for i in range(num_worker_machines)]
            if is_chief:
                if is_trainable:
                    var_update_deps = \
                        var_op_to_sync_deps[var_op] + [var_update_op]
                else:
                    var_update_deps = [var_update_op]
                # Chief enqueues tokens to all other workers
                # after executing variable update
                token = tf.constant(False)
                with tf.control_dependencies(var_update_deps):
                    for i, q in enumerate(var_update_global_sync_queues):
                        def true_fn():
                            with tf.control_dependencies([q.dequeue()]):
                                return q.enqueue(token)
                        queue_ops.append(tf.cond(q.size() > 0, true_fn, lambda: q.enqueue(token)))

        local_chief_id = worker_id - local_worker_id
        with tf.device('/job:worker/task:%d/cpu:0' % local_chief_id):
            var_update_local_sync_queues = \
                [tf.FIFOQueue(1, [tf.bool], shapes=[[]],
                              name='auto_parallel_%s_update_local_sync_queue_%d'
                                   % (var_op.name, i),
                              shared_name='auto_parallel_%s'
                                          '_update_local_sync_queue_%d'
                                          % (var_op.name, i))
                 for i in range(num_local_workers)]
            if is_local_chief:
                if is_trainable:
                    var_update_deps = \
                        var_op_to_sync_deps[var_op] \
                        + [var_update_global_sync_queues[machine_id].dequeue()]
                else:
                    var_update_deps = \
                        [var_update_global_sync_queues[machine_id].dequeue()]
                # Chief enqueues tokens to all other workers
                # after executing variable update
                token = tf.constant(False)
                with tf.control_dependencies(var_update_deps):
                    for i, q in enumerate(var_update_local_sync_queues):
                        if i != local_worker_id:
                            def true_fn():
                                with tf.control_dependencies([q.dequeue()]):
                                    return q.enqueue(token)
                            queue_ops.append(tf.cond(q.size() > 0, true_fn, lambda: q.enqueue(token)))
                        else:
                            queue_ops.append(tf.no_op())
            else:
                # wait for execution of var_update_op
                if is_trainable:
                    with tf.control_dependencies(
                            var_op_to_sync_deps[var_op]):
                        dequeue = var_update_local_sync_queues[local_worker_id]\
                            .dequeue()
                else:
                    dequeue = var_update_local_sync_queues[local_worker_id]\
                        .dequeue()
                queue_ops.append(dequeue)

            # Only dense trainable variables are replicated locally
            if master_var_op_to_mirror_vars is not None \
                    and var_op in master_var_op_to_mirror_vars:
                mirror_variable_update_ops = _get_mirror_variable_update_ops(
                    master_var_op_to_mirror_vars,
                    queue_ops,
                    trainable_var_op_to_var[var_op])
                with tf.device(this_worker_cpu):
                    finish_op = tf.group(*mirror_variable_update_ops)
            else:
                finish_op = tf.group(*queue_ops)

            if var_op == global_step_op:
                global_step_update_op = var_update_op

        with tf.device(var_op.device), tf.name_scope(""):
            if var_op == global_step_op and not is_chief:
                # Chief worker's finish_op already has update_op
                # as control input
                deps = [finish_op]
                deps.extend([inp.op for inp in var_update_op.inputs])
                deps.extend([inp for inp in var_update_op.control_inputs])
                finish_op = tf.group(*deps)
            var_op_to_finish_op[var_op] = finish_op

    # Replace variable update op with finish_op (control input)
    # or read_op (input)
    for var_op, finish_op in var_op_to_finish_op.items():
        if var_op in trainable_var_op_to_update_op:
            var_update_op = trainable_var_op_to_update_op[var_op]
        else:
            var_update_op = non_trainable_var_op_to_update_op[var_op]
        update_control_consumers(op_to_control_consumer_ops[var_update_op],
                                  var_update_op, finish_op)
        _replace_update_op_with_read_op(var_op, var_update_op, finish_op)
    var_op_to_agg_grad_ = {}
    for key in var_op_to_agg_grad:
      var_op_to_agg_grad_[key.name] = (var_op_to_agg_grad[key][0].name, var_op_to_agg_grad[key][1].name)
    trainable_var_op_to_update_op_ = {}
    for key in trainable_var_op_to_update_op:
      trainable_var_op_to_update_op_[key.name] = trainable_var_op_to_update_op[key].name    
    return var_op_to_agg_grad_, trainable_var_op_to_update_op_


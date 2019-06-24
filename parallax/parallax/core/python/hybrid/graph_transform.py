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

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training.python.training import device_setter
import horovod as hvd

from parallax.core.python.common.graph_transform_lib import *
from parallax.core.python.common.lib import *
from parallax.core.python.hybrid.in_graph_parallel import in_graph_auto_parallel_compute
from parallax.core.python.hybrid.between_graph_parallel import between_graph_auto_parallel_compute
from parallax.core.python.mpi.graph_transform import _add_aggregation_ops

def _place_post_grad_agg_ops_hybrid(ps_device,
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
                    parallax_log.debug(unique_tasks)
            if curr_op_task != SHARED:
                parallax_log.debug('post_grad_agg_op %s is assigned to %s task %d'
                              % (curr_op.name, curr_op_task[0], curr_op_task[1]))

            if curr_op_task == SHARED:
                # TODO: do not assign all SHARED ops to task 0
                # - we can do better
                curr_op_task = 0
                ps_device.task = curr_op_task
                if tf.DeviceSpec.from_string(curr_op.device).job != ps_device.job or tf.DeviceSpec.from_string(curr_op.device).task != ps_device.task: 
                  parallax_log.debug('shared_op : %s - %s -> %s' %(curr_op.name, curr_op.device, ps_device.to_string()))
                curr_op._set_device(ps_device)
                assert curr_op.device == ps_device.to_string()
            else:
                d = tf.DeviceSpec(job=curr_op_task[0], task=curr_op_task[1])
                if tf.DeviceSpec.from_string(curr_op.device).job != d.job or tf.DeviceSpec.from_string(curr_op.device).task != d.task:
                  parallax_log.debug('local_op : %s - %s -> %s' %(curr_op.name, curr_op.device, d.to_string()))
                curr_op._set_device(d)
                assert curr_op.device == d.to_string()


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
            op_to_task[agg_grad[0].op] = (var_device.job, var_device.task)
        agg_grad_ops.append(agg_grad[1].op)
        op_to_task[agg_grad[1].op] = (var_device.job, var_device.task)

    apply_grad_ops = []
    for var_op, apply_grad_op in var_op_to_apply_grad_op.items():
        var_device = tf.DeviceSpec.from_string(var_op.device)
        apply_grad_ops.append(apply_grad_op)
        # colocate apply_grad and variable
        apply_grad_op._set_device(var_device)
        op_to_task[apply_grad_op] = (var_device.job, var_device.task)

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

    parallax_log.debug('boundary_between_servers called')

    before = {}
    for op in tf.get_default_graph().get_operations():
      before[op.name] = tf.DeviceSpec.from_string(op.device)

    _assign(op_to_task, agg_grad_ops, apply_grad_ops, apply_grad_ancestor_ops,
            ancestors_diff_descendants, is_parent_to_child=True)
    _assign(op_to_task, agg_grad_ops, apply_grad_ops, apply_grad_ancestor_ops,
            ancestors_diff_descendants, is_parent_to_child=False)

    for op in tf.get_default_graph().get_operations():
      if before[op.name].job != tf.DeviceSpec.from_string(op.device).job or before[op.name].task != tf.DeviceSpec.from_string(op.device).task:
        parallax_log.debug('boundary between servers: %s, %s -> %s' % (op.name, before[op.name].to_string(), op.device))

def _add_broadcast_ops(target, worker_id):
    bcast_global_variables_ops = []
    with tf.device('/job:worker/task:%d' % worker_id):
        for var in target:
            bcast_global_variables_ops.append(
                tf.assign(var, hvd.broadcast(var, 0)))
        with tf.control_dependencies(bcast_global_variables_ops):
            tf.no_op(name='auto_parallel_bcast_global_vars')


def graph_transform_dense_mpi(worker_id,
                              meta_graph_def,
                              op_library_path,
                              config):
    with tf.Graph().as_default() as graph:
        tf.train.import_meta_graph(meta_graph_def)

        num_workers = hvd.size()

        op_to_control_consumer_ops = get_all_control_consumers(graph)
        trainable_variable_ops = [var.op for var in tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)]

        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        local_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)

        sparse_var_ops = \
            set([gradients_info._target.op
             for gradients_info in tf.get_collection(tf.GraphKeys.GRADIENTS_INFO)
             if not isinstance(gradients_info._grad, tf.Tensor)])

        for op in tf.get_default_graph().get_operations():
            if op.type in sparse_var_update_op_types.keys():
                sparse_var_ops.add(op.inputs[UPDATE_OP_VAR_POS].op)

        global_grad_ops = [var.op
                           for var in tf.get_collection(PARALLAX_GLOBAL_GRADS)]

        for var in global_variables + local_variables:
            op = var.op
            if op not in sparse_var_ops and op not in global_grad_ops:
                op._set_device('/job:worker/task:%d' % worker_id)

        var_op_to_agg_grad = {}
        with tf.device('/job:worker/task:%d' % worker_id):
            for gradients_info in tf.get_collection(tf.GraphKeys.GRADIENTS_INFO):
                target_tensor = gradients_info._target
                grad = gradients_info._grad
                if target_tensor.op not in trainable_variable_ops:
                    parallax_log.debug(
                        "Gradient for non-trainable variable %s is created, ignore"
                        % target_tensor.op.name)
                    continue
                if isinstance(grad, tf.Tensor):
                    _add_aggregation_ops(gradients_info, op_to_control_consumer_ops,
                                         config)
                    # Now, gradients_info._grad = aggregated gradient
                    var_op_to_agg_grad[target_tensor.op.name] = gradients_info._grad.name

    return tf.train.export_meta_graph(graph=graph), var_op_to_agg_grad


def _init_for_worker_vars(meta_graph_def, worker_id):
    with tf.Graph().as_default() as graph:
        tf.train.import_meta_graph(meta_graph_def)
        worker_vars = []
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        local_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        for var in global_variables + local_variables:
            device = tf.DeviceSpec.from_string(var.device)
            if device.job == 'worker' and device.task == worker_id:
                worker_vars.append(var)
        _add_broadcast_ops(worker_vars, worker_id)
        meta_graph_def = tf.train.export_meta_graph(graph=graph)
        return meta_graph_def

def graph_transform_hybrid(single_gpu_meta_graph_def,
                           worker_id,
                           local_worker_id,
                           machine_id,
                           hostname,
                           config):
    cluster_info = config.resource_info
    this_worker = None
    for w in cluster_info['worker']:
        if w['hostname'] == hostname:
            this_worker = w
    num_worker_machines = len(cluster_info['worker'])
    num_local_workers = max(len(this_worker['gpus']), 1)
    ps_device = '/job:ps' if 'ps' in cluster_info else '/job:worker/cpu:0'
    ps_job_name = tf.DeviceSpec.from_string(ps_device).job
    cluster_spec = get_tf_clusterspec(cluster_info).as_dict()
    num_pss = len(cluster_spec[ps_job_name])

    cluster_spec = get_tf_clusterspec(cluster_info)

    parallax_log.debug(
        "Starting graph transformation for PS for worker %d" % worker_id)

    # TODO: handle op_library_path
    tensor_or_op_name_to_replica_names = TensorOrOpNameToReplicaNames(
        single_gpu_meta_graph_def.meta_info_def.stripped_op_list)

    multi_gpu_meta_graph_def = \
        in_graph_auto_parallel_compute(
            single_gpu_meta_graph_def, worker_id, local_worker_id,
            num_local_workers, config, None,
            tensor_or_op_name_to_replica_names)

    local_aggregation = config.communication_config.ps_config.local_aggregation
    ps_meta_graph_def, var_op_to_agg_grad, trainable_var_op_to_update_op = \
        between_graph_auto_parallel_compute(
            multi_gpu_meta_graph_def,
            worker_id=worker_id,
            local_worker_id=local_worker_id,
            machine_id=machine_id,
            num_local_workers=num_local_workers,
            num_worker_machines=num_worker_machines,
            ps_device=ps_device,
            worker_device='/job:worker/task:%d' % worker_id,
            merge_devices=True,
            cluster_spec=cluster_spec,
            sync=True,
            op_library_path=None,
            num_replicas_per_worker=1,
            average_sparse=config.average_sparse,
            tensor_or_op_name_to_replica_names=tensor_or_op_name_to_replica_names,
            local_aggregation=local_aggregation)

    meta_graph_def, dense_var_op_to_agg_grad = \
        graph_transform_dense_mpi(worker_id,
                                  ps_meta_graph_def,
                                  op_library_path=None,
                                  config=config)

    with tf.Graph().as_default() as graph:
        tf.train.import_meta_graph(meta_graph_def)
        var_op_to_agg_grad_ = {}
        for var_op_name in var_op_to_agg_grad: 
          var_op = graph.get_operation_by_name(var_op_name)
          indices = graph.get_tensor_by_name(var_op_to_agg_grad[var_op_name][0])
          values = graph.get_tensor_by_name(var_op_to_agg_grad[var_op_name][1])
          var_op_to_agg_grad_[var_op] = (indices, values)

        trainable_var_op_to_update_op_ = {}     
        for var_op_name in trainable_var_op_to_update_op:
          var_op = graph.get_operation_by_name(var_op_name)
          update_op = graph.get_operation_by_name(trainable_var_op_to_update_op[var_op_name])
          trainable_var_op_to_update_op_[var_op] = update_op 

        # TODO If we do MPI transform first, we can bring this block back to
        # between-graph transform.
        for var_op_name in dense_var_op_to_agg_grad.keys():
           var_op = graph.get_operation_by_name(var_op_name)
           grad =  graph.get_tensor_by_name(dense_var_op_to_agg_grad[var_op_name])
           if var_op in var_op_to_agg_grad_:
                raise RuntimeError('A dense variable is handled by between-graph.')
           else:
                var_op_to_agg_grad_[var_op] = \
                    (None, grad)

        if config.communication_config.ps_config.boundary_among_servers:
            _place_post_grad_agg_ops_hybrid(tf.DeviceSpec.from_string(ps_device),
                                            var_op_to_agg_grad_, trainable_var_op_to_update_op_)
        if config.communication_config.ps_config.boundary_between_workers_and_servers:
            set_boundary_between_workers_and_servers()
        meta_graph_def = tf.train.export_meta_graph(graph=graph)

    meta_graph_def = _init_for_worker_vars(meta_graph_def, worker_id)

    parallax_log.debug(
        "Finished graph transformation for PS for worker %d" % worker_id)

    return meta_graph_def, tensor_or_op_name_to_replica_names.export()

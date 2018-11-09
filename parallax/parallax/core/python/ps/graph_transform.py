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

from parallax.core.python.common.lib import *
from parallax.core.python.ps.in_graph_parallel import in_graph_auto_parallel_compute
from parallax.core.python.ps.between_graph_parallel import between_graph_auto_parallel_compute


def graph_transform_ps(single_gpu_meta_graph_def,
                       worker_id,
                       config,
                       op_library_path=None):
    cluster_info = config.resource_info
    # TODO: Handle all ps configurations 
    if config.communication_config.ps_config.replicate_variables and not config.sync:
        raise ValueError('replicate_variables is only possible with sync')
    ps_device = '/job:ps' if 'ps' in cluster_info else '/job:worker/cpu:0'
    cluster_spec = get_tf_clusterspec(cluster_info)
    worker = cluster_info['worker'][worker_id]
    num_replicas_per_worker = max(1, len(worker['gpus']))

    parallax_log.debug(
        "Starting graph transformation for PS for worker %d" % worker_id)

    tensor_or_op_name_to_replica_names = TensorOrOpNameToReplicaNames(
        single_gpu_meta_graph_def.meta_info_def.stripped_op_list)

    multi_gpu_meta_graph_def = \
        in_graph_auto_parallel_compute(
            single_gpu_meta_graph_def, num_gpus, config=config,
            op_library_path=op_library_path,
            tensor_or_op_name_to_replica_names=tensor_or_op_name_to_replica_names)

    ps_meta_graph_def = \
        between_graph_auto_parallel_compute(
            multi_gpu_meta_graph_def,
            worker_id=worker_id,
            ps_device=ps_device,
            worker_device='/job:worker/task:%d' % worker_id,
            merge_devices=True,
            cluster_spec=cluster_spec,
            config=config,
            op_library_path=op_library_path,
            num_replicas_per_worker=num_replicas_per_worker,
            tensor_or_op_name_to_replica_names=tensor_or_op_name_to_replica_names)
    parallax_log.debug(
        "Finished graph transformation for PS for worker %d" % worker_id)
    return ps_meta_graph_def, tensor_or_op_name_to_replica_names.export()

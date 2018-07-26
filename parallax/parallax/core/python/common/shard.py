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


NUM_SHARDS = "num_shards"
SHARD_ID = "shard_id"
SHARD_FILTER_PRED = "shard_filter_predicate"
FILTER_DATASET_NUM_SHARDS_POS = 1
FILTER_DATASET_SHARD_ID_POS = 2


def create_num_shards_and_shard_id():
    """Returns and create the num shards and the shard id tensors.

    Returns:
      The num shards and the shard id tensors.

    Raises:
      ValueError: if the num shards tensor or the shard id tensor is already
      defined.
    """

    # TODO: allow num_shards and shard_id inside a library function
    graph = tf.get_default_graph()
    num_shards_tensors = graph.get_collection(NUM_SHARDS)
    if len(num_shards_tensors) > 0:
        raise ValueError('"num_shards" already exists.')
    shard_id_tensors = graph.get_collection(SHARD_ID)
    if len(shard_id_tensors) > 0:
        raise ValueError('"shard_id" already exists.')
    # Create in proper graph and base name_scope.
    with graph.as_default() as g, g.name_scope(None):
        # Initialize num_shards_tensor=1, and shard_id_tensor=0.
        # parallax updates the value when the graph is transformed
        # for distributed version.
        num_shards_tensor = tf.constant(1, dtype=tf.int64, name="num_shards")
        shard_id_tensor = tf.constant(0, dtype=tf.int64, name="shard_id")
    tf.add_to_collection(NUM_SHARDS, num_shards_tensor)
    tf.add_to_collection(SHARD_ID, shard_id_tensor)
    return num_shards_tensor, shard_id_tensor


def _get_or_create_num_shards_and_shard_id():
    graph = tf.get_default_graph()
    num_shards_tensors = graph.get_collection(NUM_SHARDS)
    if len(num_shards_tensors) > 0:
        num_shards_tensor = num_shards_tensors[0]
        shard_id_tensor = \
            graph.get_collection(SHARD_ID)[0]
    else:
        num_shards_tensor, shard_id_tensor = create_num_shards_and_shard_id()
    return num_shards_tensor, shard_id_tensor


def shard(ds):
    """Convert a dataset to include shard, it has same effect
    with ds.shard(num_shards, index).
    """

    # TODO: allow dataset shard inside a function or dataset api
    # (e.g., map, parallel_interleave)
    num_shards, shard_id = _get_or_create_num_shards_and_shard_id()

    def filter_fn(elem_index, _):
        mod_result = tf.mod(elem_index, num_shards)
        return tf.equal(mod_result, shard_id)

    f = ds._enumerate().filter(filter_fn)
    assert f._predicate.captured_inputs[0] == num_shards
    assert f._predicate.captured_inputs[1] == shard_id
    tf.add_to_collection(SHARD_FILTER_PRED,
                         f._predicate.name)
    return f.map(lambda _, elem: elem)

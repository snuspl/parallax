# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


import tensorflow as tf
from parallax import shard

# A SentenceBatch is a pair of Tensors:
#  ids: Batch of input sentences represented as sequences of word ids: an int64
#    Tensor with shape [batch_size, padded_length].
#  mask: Boolean mask distinguishing real words (1) from padded words (0): an
#    int32 Tensor with shape [batch_size, padded_length].
SentenceBatch = collections.namedtuple("SentenceBatch", ("ids", "mask"))


def parse_example_batch(serialized):
  """Parses a batch of tf.Example protos.

  Args:
    serialized: A 1-D string Tensor; a batch of serialized tf.Example protos.
  Returns:
    encode: A SentenceBatch of encode sentences.
    decode_pre: A SentenceBatch of "previous" sentences to decode.
    decode_post: A SentenceBatch of "post" sentences to decode.
  """
  features = tf.parse_example(
      serialized,
      features={
          "encode": tf.VarLenFeature(dtype=tf.int64),
          "decode_pre": tf.VarLenFeature(dtype=tf.int64),
          "decode_post": tf.VarLenFeature(dtype=tf.int64),
      })

  def _sparse_to_batch(sparse):
    ids = tf.sparse_tensor_to_dense(sparse)  # Padding with zeroes
    mask = tf.sparse_to_dense(sparse.indices, sparse.dense_shape,
                              tf.ones_like(sparse.values, dtype=tf.int32))
    return SentenceBatch(ids=ids, mask=mask)

  output_names = ("encode", "decode_pre", "decode_post")
  ret = [_sparse_to_batch(features[x]) for x in output_names]
  input_size = tf.size(features['encode'].indices)
  ret.append(input_size)
  return tuple(ret)


def prefetch_input_data(reader,
                        file_pattern,
                        shuffle,
                        capacity,
                        num_reader_threads=1):
  """Prefetches string values from disk into an input queue.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        "/tmp/train_data-?????-of-00100", where '?' acts as a wildcard that
        matches any character).
    shuffle: Boolean; whether to randomly shuffle the input data.
    capacity: Queue capacity (number of records).
    num_reader_threads: Number of reader threads feeding into the queue.

  Returns:
    A Queue containing prefetched string values.
  """
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)
  data_files.sort()
  num_files = len(data_files)
  num_shards, shard_id = shard.create_num_shards_and_shard_id()
  shard_size = num_files / num_shards
  shard_size = tf.cast(shard_size, dtype=tf.int64)
  remainder = num_files % num_shards
  
  slice_begin = tf.cond(tf.less(shard_id, remainder + 1),
                        lambda: (shard_size + 1) * shard_id,
                        lambda: shard_size * shard_id + remainder)
  slice_size = tf.cond(tf.less(shard_id, remainder), lambda: shard_size + 1,
                       lambda: shard_size)
  data_files = tf.slice(data_files, [slice_begin], [slice_size])
  filename_queue = tf.train.string_input_producer(
      data_files, shuffle=shuffle, capacity=16, name="filename_queue")

  if shuffle:
    min_after_dequeue = int(0.6 * capacity)
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        dtypes=[tf.string],
        shapes=[[]],
        name="random_input_queue")
  else:
    values_queue = tf.FIFOQueue(
        capacity=capacity,
        dtypes=[tf.string],
        shapes=[[]],
        name="fifo_input_queue")

  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(
      tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops))
  tf.summary.scalar("queue/%s/fraction_of_%d_full" % (values_queue.name,
                                                      capacity),
                    tf.cast(values_queue.size(), tf.float32) * (1.0 / capacity))

  return values_queue

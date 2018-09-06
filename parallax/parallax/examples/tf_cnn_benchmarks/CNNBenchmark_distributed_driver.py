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

import argparse
import sys
import os
import json
import time
import numpy as np

from absl import flags
import tensorflow as tf

import benchmark_cnn
import cnn_util
import parallax_config
from cnn_util import log_fn
from tensorflow.core.protobuf import config_pb2

import parallax

benchmark_cnn.define_flags()
flags.adopt_module_key_flags(benchmark_cnn)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('resource_info_file',
                           os.path.abspath(os.path.join(
                               os.path.dirname(__file__),
                               '.',
                               'resource_info')),
                           'Filename containing cluster information')
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of iterations to run for each workers.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How many steps between two runop logs.""")
tf.app.flags.DEFINE_boolean('sync', True, '')

def main(_):

    # Build benchmark_cnn model
    params = benchmark_cnn.make_params_from_flags()
    params, sess_config = benchmark_cnn.setup(params)
    bench = benchmark_cnn.BenchmarkCNN(params)

    # Print informaton
    tfversion = cnn_util.tensorflow_version_tuple()
    log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))
    bench.print_info()

    # Build single-GPU benchmark_cnn model
    single_gpu_graph = tf.Graph()
    with single_gpu_graph.as_default():
        bench.build_model()
        summary = tf.summary.scalar('train_loss', bench.cost)
        train_writer = tf.summary.FileWriter('/home/soojeong/resnet50_tensorboard')

    config = parallax_config.build_config()
    config.sess_config = sess_config
    sess, num_workers, worker_id, num_replicas_per_worker = \
        parallax.parallel_run(single_gpu_graph,
                              FLAGS.resource_info_file,
                              sync=FLAGS.sync,
                              parallax_config=config)

    fetches = {
        'global_step': bench.global_step,
        'cost': bench.cost,
        'train_op': bench.train_op,
        'summary': summary
    }

    start = time.time()
    for i in range(FLAGS.max_steps):
        results = sess.run(fetches)
        if i % FLAGS.log_frequency == 0:
            train_writer.add_summary(results['summary'][0], i)
            end = time.time()
            throughput = float(FLAGS.log_frequency) / float(end - start)
            parallax.log.info(
                "global step: %d, loss: %f, throughput: %f steps/sec"
                % (results['global_step'][0], results['cost'][0], throughput))
            start = time.time()

if __name__ == '__main__':
    tf.app.run()

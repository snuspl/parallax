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
import os
import json
import time

import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import parallax
import parallax_config

import configuration
import skip_thoughts_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_path', None,
                           """Where to training/test data is stored.""")
tf.app.flags.DEFINE_string('input_file_pattern', '',
                           """File pattern of train data""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch_size""")
tf.app.flags.DEFINE_string('resource_info_file',
                           os.path.abspath(
                               os.path.join(os.path.dirname(__file__), '.',
                                            'resource_info')),
                           'Filename containing cluster information')
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of iterations to run for each workers.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How many steps between two runop logs.""")
tf.app.flags.DEFINE_boolean('sync', True, '')

def main(_):
    single_gpu_graph = tf.Graph()
    with single_gpu_graph.as_default():
        model_config = configuration.model_config(
            input_file_pattern=FLAGS.input_file_pattern,
            batch_size=FLAGS.batch_size)
        training_config = configuration.training_config()
        model = skip_thoughts_model.SkipThoughtsModel(model_config,
                                                      mode="train")
        model.build()

        # Setup learning rate
        if training_config.learning_rate_decay_factor > 0:
            learning_rate = tf.train.exponential_decay(
                learning_rate=float(training_config.learning_rate),
                global_step=model.global_step,
                decay_steps=training_config.learning_rate_decay_steps,
                decay_rate=training_config.learning_rate_decay_factor,
                staircase=False)
        else:
            learning_rate = tf.constant(training_config.learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        train_tensor = tf.contrib.slim.learning.create_train_op(
            total_loss=model.total_loss,
            optimizer=optimizer,
            global_step=model.global_step,
            clip_gradient_norm=training_config.clip_gradient_norm)
        print("# Trainable variables")
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for param in params:
            print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                    param.op.device)) 
    def run(sess, num_workers, worker_id, num_replicas_per_worker):
        fetches = {
            'global_step':
                model.global_step,
            'cost':
                model.total_loss,
            'train_op':
                train_tensor,
            'input_size':
                model.input_size
        }

        start = time.time()
        words = 0
        for i in range(FLAGS.max_steps):
            results = sess.run(fetches)
            words += np.sum(results['input_size'])

            if i % FLAGS.log_frequency == 0:
                end = time.time()
                wps = words / float(end - start)
                throughput = float(FLAGS.log_frequency) / float(end - start)
                parallax.log.info(
                    "global step: %d, loss: %f, throughput: %f steps/sec, wps: %f"
                    % (results['global_step'][0], results['cost'][0], throughput, wps))
                start = time.time()
                words = 0

    sess, num_workers, worker_id, num_replicas_per_worker = \
        parallax.parallel_run(single_gpu_graph,
                              FLAGS.resource_info_file,
                              sync=FLAGS.sync,
                              parallax_config=parallax_config.build_config())
    run(sess, num_workers, worker_id, num_replicas_per_worker)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

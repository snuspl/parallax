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
import parallax


import os
import time
import math
import json
import sys
import numpy as np

from data_utils import Vocabulary, Dataset
import language_model_graph
import parallax_config

flags = tf.app.flags
flags.DEFINE_string("logdir", "/tmp/lm1b", "Logging directory.")
flags.DEFINE_string("datadir", None, "Logging directory.")
flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
flags.DEFINE_integer("eval_steps", 70, "Number of eval steps.")
flags.DEFINE_string('resource_info_file',
                    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 '.',
                                                 'resource_info')),
                    'Filename containing cluster information')
flags.DEFINE_integer('max_steps', 1000000,
                     """Number of iterations to run for each workers.""")
flags.DEFINE_integer('log_frequency', 100,
                     """How many steps between two runop logs.""")
flags.DEFINE_boolean('sync', True, '')
FLAGS = flags.FLAGS


def main(_):

    vocab = Vocabulary.from_file(os.path.join(FLAGS.datadir, "1b_word_vocab.txt"))
    dataset = Dataset(vocab, os.path.join(FLAGS.datadir, "training-monolingual.tokenized.shuffled/*"))

    with tf.Graph().as_default() as single_gpu_graph:
        with tf.variable_scope("model"):
            model = language_model_graph.build_model()

    state_c = []
    state_h = []

    def run(sess, num_iters, tensor_or_op_name_to_replica_names,
            num_workers, worker_id, num_replicas_per_worker):

        if len(state_c) == 0:
            state_c.extend([np.zeros([FLAGS.batch_size, model.state_size], dtype=np.float32) for _ in range(num_replicas_per_worker)])
            state_h.extend([np.zeros([FLAGS.batch_size, model.projected_size], dtype=np.float32) for _ in range(num_replicas_per_worker)])

        prev_global_step = sess.run(tensor_or_op_name_to_replica_names[model.global_step.name][0])
        prev_time = time.time()
        data_iterator = dataset.iterate_forever(FLAGS.batch_size * num_replicas_per_worker,
                                                FLAGS.num_steps, num_workers, worker_id)
        fetches = {
            'global_step': tensor_or_op_name_to_replica_names[model.global_step.name][0],
            'loss': tensor_or_op_name_to_replica_names[model.loss.name][0],
            'train_op': tensor_or_op_name_to_replica_names[model.train_op.name][0]
        }
        for replica_id in range(num_replicas_per_worker):
            fetches['final_state_c_%d' % replica_id] = tensor_or_op_name_to_replica_names[model.final_state_c.name][replica_id]
            fetches['final_state_h_%d' % replica_id] = tensor_or_op_name_to_replica_names[model.final_state_h.name][replica_id]
        x_names = tensor_or_op_name_to_replica_names[model.x.name]
        y_names = tensor_or_op_name_to_replica_names[model.y.name]
        w_names = tensor_or_op_name_to_replica_names[model.w.name]
        state_c_names = tensor_or_op_name_to_replica_names[model.initial_state_c.name]
        state_h_names = tensor_or_op_name_to_replica_names[model.initial_state_h.name]
        for local_step in range(num_iters):
            if FLAGS.use_synthetic:
              x = np.random.randint(low=0, high=model.vocab_size, size=(FLAGS.batch_size*num_replicas_per_worker, FLAGS.num_steps))
              y = np.random.randint(low=0, high=model.vocab_size, size=(FLAGS.batch_size*num_replicas_per_worker, FLAGS.num_steps))
              w = np.ones((FLAGS.batch_size*num_replicas_per_worker, FLAGS.num_steps))
            else:
              x, y, w = next(data_iterator)
            feeds = {}
            for replica_id in range(num_replicas_per_worker):
                start_idx = FLAGS.batch_size * replica_id
                end_idx = FLAGS.batch_size * (replica_id + 1)
                feeds[x_names[replica_id]] = x[start_idx:end_idx]
                feeds[y_names[replica_id]] = y[start_idx:end_idx]
                feeds[w_names[replica_id]] = w[start_idx:end_idx]
                feeds[state_c_names[replica_id]] = state_c[replica_id]
                feeds[state_h_names[replica_id]] = state_h[replica_id]
            fetched = sess.run(fetches, feeds)

            for replica_id in range(num_replicas_per_worker):
                state_c[replica_id] = fetched['final_state_c_%d' % replica_id]
                state_h[replica_id] = fetched['final_state_h_%d' % replica_id]

            if local_step % FLAGS.log_frequency == 0:
                cur_time = time.time()
                elapsed_time = cur_time - prev_time
                num_words = FLAGS.batch_size * FLAGS.num_steps
                wps = (fetched['global_step'] - prev_global_step) * num_words / elapsed_time
                prev_global_step = fetched['global_step']
                parallax.log.info("Iteration %d, time = %.2fs, wps = %.0f, train loss = %.4f" % (
                    fetched['global_step'], cur_time - prev_time, wps, fetched['loss']))
                prev_time = cur_time

    parallax.parallel_run(single_gpu_graph,
                          run,
                          FLAGS.resource_info_file,
                          FLAGS.max_steps,
                          sync=FLAGS.sync,
                          parallax_config=parallax_config.build_config())

if __name__ == "__main__":
    tf.app.run()

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
flags.DEFINE_boolean('deterministic', False, '')
flags.DEFINE_integer('vocab_size_limit', 793470, '')
flags.DEFINE_integer('emb_size', 512, '')
FLAGS = flags.FLAGS


def main(_):

    vocab = Vocabulary.from_file(os.path.join('/cmsdata/ssd1/cmslab/lm1b', "1b_word_vocab.txt"), FLAGS.vocab_size_limit)
    dataset = Dataset(vocab, os.path.join(FLAGS.datadir, "*"), FLAGS.deterministic)

    single_gpu_graph = tf.Graph()
    with single_gpu_graph.as_default():
        with tf.variable_scope("model"):
            model = language_model_graph.build_model()

    def run(sess,num_workers, worker_id, num_replicas_per_worker):
        
        state_c = []
        state_h = []

        if len(state_c) == 0:
            state_c.extend([np.zeros([FLAGS.batch_size, model.state_size], dtype=np.float32) for _ in range(num_replicas_per_worker)])
            state_h.extend([np.zeros([FLAGS.batch_size, model.projected_size], dtype=np.float32) for _ in range(num_replicas_per_worker)])

        prev_global_step = sess.run(model.global_step)[0]
        prev_time = time.time()
        data_iterator = dataset.iterate_forever(FLAGS.batch_size * num_replicas_per_worker,
                                                FLAGS.num_steps, num_workers, worker_id)
        fetches = {
            'global_step': model.global_step,
            'loss': model.loss,
            'train_op': model.train_op,
            'final_state_c': model.final_state_c,
            'final_state_h': model.final_state_h
        }

        for local_step in range(FLAGS.max_steps):
            if FLAGS.use_synthetic:
              x = np.random.randint(low=0, high=model.vocab_size, size=(FLAGS.batch_size*num_replicas_per_worker, FLAGS.num_steps))
              y = np.random.randint(low=0, high=model.vocab_size, size=(FLAGS.batch_size*num_replicas_per_worker, FLAGS.num_steps))
              w = np.ones((FLAGS.batch_size*num_replicas_per_worker, FLAGS.num_steps))
            else:
              x, y, w = next(data_iterator)
            feeds = {}
            feeds[model.x] = np.split(x, num_replicas_per_worker)
            feeds[model.y] = np.split(y, num_replicas_per_worker)
            feeds[model.w] = np.split(w, num_replicas_per_worker)
            feeds[model.initial_state_c] = state_c
            feeds[model.initial_state_h] = state_h
            fetched = sess.run(fetches, feeds)

            state_c = fetched['final_state_c']
            state_h = fetched['final_state_h']

            if local_step % FLAGS.log_frequency == 0:
                cur_time = time.time()
                elapsed_time = cur_time - prev_time
                num_words = FLAGS.batch_size * FLAGS.num_steps
                wps = (fetched['global_step'][0] - prev_global_step) * num_words / elapsed_time
                prev_global_step = fetched['global_step'][0]
                parallax.log.info("Iteration %d, time = %.2fs, wps = %.0f, train loss = %.4f" % (
                    fetched['global_step'][0], cur_time - prev_time, wps, fetched['loss'][0]))
                prev_time = cur_time

    sess, num_workers, worker_id, num_replicas_per_worker = \
        parallax.parallel_run(single_gpu_graph,
                              FLAGS.resource_info_file,
                              sync=FLAGS.sync,
                              parallax_config=parallax_config.build_config())
    run(sess, num_workers, worker_id, num_replicas_per_worker)
    os.system('kill %d' % os.getpid())

if __name__ == "__main__":
    tf.app.run()

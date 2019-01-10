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

    vocab = Vocabulary.from_file(os.path.join('/cmsdata/ssd1/cmslab/lm1b', "1b_word_vocab.txt"))
    dataset = Dataset(vocab, os.path.join(FLAGS.datadir, "*"))

    single_gpu_graph = tf.Graph()
    with single_gpu_graph.as_default():
        with tf.variable_scope("model"):
            model = language_model_graph.build_model()

    data_iterator = dataset.iterate_once(FLAGS.batch_size,
                                            FLAGS.num_steps)
    unique_size = 0
    step = 0
    while True:
        x, y, w = next(data_iterator)
        unique_size += len(np.unique(x))
        step += 1
        if step % 50 == 0:
            print(unique_size / step)
            sys.stdout.flush()

if __name__ == "__main__":
    tf.app.run()

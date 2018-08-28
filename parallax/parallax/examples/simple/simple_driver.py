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

import os
import numpy as np
import tensorflow as tf
import argparse

import parallax

parser = argparse.ArgumentParser()
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01,
                    help='Learning rate')

args = parser.parse_args()

[-0.880728357, -0.706550564],
[-0.179175969, 0.052373456],
[0.460992645, 0.328267666],
[-0.378916048, 0.86581809],
[-0.064562793, -0.755948805],
[-0.585833517, -0.46743004],
[-0.151177544, -0.582325109],
[-0.720116833, 0.834904979],
[-0.518939078, -0.670627318],
[-0.035878422, 0.750102543],
[-0.673400627, -0.919498322],
[-0.731202767, -0.159733489],
[-0.463404605, 0.697764632],
[0.706744043, 0.458026442],
[0.819940015, -0.867168658],
[-0.056113501, -0.602024627],
[0.213450484, -0.20133007],
[-0.358544296, -0.40380244],

train_x = np.array([
    [-0.880728357, -0.706550564],
    [-0.179175969, 0.052373456],
    [0.460992645, 0.328267666],
    [-0.378916048, 0.86581809],
    [-0.064562793, -0.755948805],
    [-0.585833517, -0.46743004],
    [-0.151177544, -0.582325109],
    [-0.720116833, 0.834904979],
    [-0.518939078, -0.670627318],
    [-0.035878422, 0.750102543],
    [-0.673400627, -0.919498322],
    [-0.731202767, -0.159733489],
    [-0.463404605, 0.697764632],
    [0.706744043, 0.458026442],
    [0.819940015, -0.867168658],
    [-0.056113501, -0.602024627],
    [0.213450484, -0.20133007],
    [-0.358544296, -0.40380244]
])

train_y = np.array([
    [2.306799664],
    [1.825970013],
    [1.901374447],
    [0.909895597],
    [2.723102683],
    [2.145410027],
    [2.498034199],
    [0.844066487],
    [2.401599333],
    [1.274285598],
    [2.542184193],
    [1.81653423],
    [1.06511757],
    [1.891457798],
    [3.317388286],
    [2.579920223],
    [2.301286159],
    [2.197386858],
])

num_samples = train_x.shape[0]


def main(_):
  with tf.Graph().as_default() as single_gpu_graph:
    global_step = tf.train.get_or_create_global_step()
    x = tf.placeholder(tf.float32, shape=(2))
    y = tf.placeholder(tf.float32, shape=(1))

    w = tf.get_variable(name='w', shape=(2, 1))
    b = tf.get_variable(name='b', shape=(1))

    pred = tf.nn.bias_add(tf.matmul(tf.expand_dims(x, axis=0), w), b)
    loss = tf.reduce_sum(tf.pow(pred - tf.expand_dims(y, axis=0), 2)) / 2

    optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # init = tf.global_variables_initializer()

  def run(sess, num_workers, worker_id, num_replicas_per_worker):
    cursor = 0
    for i in range(1000):
      feed_dict = {}
      feed_dict[x] = [train_x[(cursor + j) % num_samples] for j in \
          range(num_replicas_per_worker)]
      feed_dict[y] = [train_y[(cursor + j) % num_samples] for j in \
          range(num_replicas_per_worker)]
      cursor += num_replicas_per_worker
      fetches = {
          'global_step': global_step,
          'loss': loss,
          'train_op': train_op
      }

      results = sess.run(fetches, feed_dict=feed_dict)

      if i % 5 == 0:
        print("global step: %d, loss: %f"
              % (results['global_step'][0], results['loss'][0]))

  resource_info = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'resource_info')
  sess, num_workers, worker_id, num_replicas_per_worker = \
      parallax.parallel_run(single_gpu_graph, resource_info)
  run(sess, num_workers, worker_id, num_replicas_per_worker)

if __name__ == '__main__':
  tf.app.run()

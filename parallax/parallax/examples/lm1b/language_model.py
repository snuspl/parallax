from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base

import parallax

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('num_variable_shards', 32, 'Number of variable shard')

class LM(base.Layer):
  def __init__(self, num_steps):
    super(LM, self).__init__()
    self.num_steps = num_steps
    self.num_shards = FLAGS.num_variable_shards
    # Use keep_prob 1.0 at evaluation
    self.keep_prob = 0.9

    self.vocab_size = 793470
    self.emb_size = 512
    self.state_size = 2048
    self.projected_size = 512
    # Use num_sampled 0 (full softmax) at evaluation
    self.num_sampled = 8192

  def build(self, input_shape):
    partitioner = parallax.get_partitioner(self.num_shards)
    with tf.variable_scope(tf.get_variable_scope(), partitioner=partitioner):
      self.emb = tf.get_variable('emb', 
                                 shape=[self.vocab_size, self.emb_size],
                                 initializer=tf.uniform_unit_scaling_initializer(),
                                 trainable=True,
                                 dtype=tf.float32)
      self.softmax_w = tf.get_variable(name='softmax_w',
                                       shape=[self.vocab_size, self.projected_size],
                                       initializer=tf.uniform_unit_scaling_initializer(),
                                       trainable=True,
                                       dtype=tf.float32)

    self.softmax_b = self.add_variable(name='softmax_b',
                                       shape=[self.vocab_size],
                                       trainable=True,
                                       dtype=tf.float32)
    self.W = self.add_variable(name='W',
                               shape=[self.emb_size + self.projected_size, 4 * self.state_size],
                               trainable=True,
                               dtype=tf.float32)
    self.B = self.add_variable(name='B',
                               shape=[4 * self.state_size],
                               trainable=True,
                               dtype=tf.float32)
    self.W_P = self.add_variable(name='W_P',
                                 shape=[self.state_size, self.projected_size],
                                 trainable=True,
                                 dtype=tf.float32)
    self.built = True

  def call(self, x, y, w, initial_state_c, initial_state_h, training):
    # [bs, steps, emb_size]
    x = tf.nn.embedding_lookup(self.emb, x)
    if training:
      x = tf.nn.dropout(x, self.keep_prob)

    # [bs, emb_size] * steps
    inputs = [tf.squeeze(v, axis=[1]) for v in tf.split(value=x, num_or_size_splits=self.num_steps, axis=1)]

    c = initial_state_c
    h = initial_state_h
    for t in range(self.num_steps):
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      cell_inputs = tf.concat([inputs[t], h], axis=1)
      lstm_matrix = tf.nn.xw_plus_b(cell_inputs, self.W, self.B)
      i, j, f, o = tf.split(lstm_matrix, 4, axis=1)

      c = tf.sigmoid(f + 1.0) * c + tf.sigmoid(i) * tf.tanh(j)
      h = tf.sigmoid(o) * tf.tanh(c)
      h = tf.matmul(h, self.W_P)
      inputs[t] = h
      if training:
        inputs[t] = tf.nn.dropout(inputs[t], self.keep_prob)

    inputs[t] = tf.identity(inputs[t])

    inputs = tf.reshape(tf.concat(inputs, axis=1), [-1, self.projected_size])

    if training:
      targets = tf.reshape(y, [-1, 1])
      loss = tf.nn.sampled_softmax_loss(self.softmax_w,
                                        self.softmax_b,
                                        targets,
                                        inputs,
                                        self.num_sampled,
                                        self.vocab_size)
    else:
      full_softmax_w = tf.reshape(tf.concat(self.softmax_w, axis=1), [-1, self.projected_size])
      full_softmax_w = full_softmax_w[:self.vocab_size, :]

      logits = tf.matmul(inputs, full_softmax_w, transpose_b=True) + self.softmax_b
      targets = tf.reshape(y, [-1])
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

    loss = tf.reduce_mean(loss * tf.reshape(tf.to_float(w), [-1]))
    return loss, c, h

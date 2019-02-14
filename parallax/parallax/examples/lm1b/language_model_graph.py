from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import language_model

_NUM_WORDS = {
    'train': 798945280,
    'validation': 7789987,
}

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.flags.DEFINE_integer('num_steps', 20, 'Number of steps')
tf.flags.DEFINE_float('learning_rate', 0.2, 'Learning rate')
tf.flags.DEFINE_float('max_grad_norm', 10.0, 'max_grad_norm')
tf.flags.DEFINE_integer('num_epoch', 5, 'Number of epoch')
tf.flags.DEFINE_boolean('use_synthetic', False, 'whether to use synthetic data or not')


def build_model():
  model = language_model.LM(FLAGS.num_steps)
  global_step = tf.train.get_or_create_global_step()

  with tf.device('/gpu:0'):
    placeholder_x = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.num_steps])
    placeholder_y = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.num_steps])
    placeholder_w = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.num_steps])
    initial_state_c = tf.placeholder(dtype=tf.float32,
                                     shape=[FLAGS.batch_size, model.state_size],
                                     name='initial_c')
    initial_state_h = tf.placeholder(dtype=tf.float32,
                                     shape=[FLAGS.batch_size, model.projected_size],
                                     name='initial_h')
    loss, final_state_c, final_state_h = model(placeholder_x, placeholder_y, placeholder_w, initial_state_c, initial_state_h, training=True)
    scaled_loss = loss * FLAGS.num_steps

    emb_vars = list(model.emb)
    lstm_vars = [model.W, model.B, model.W_P]
    softmax_vars = list(model.softmax_w) + [model.softmax_b]
    all_vars = emb_vars + lstm_vars + softmax_vars
    grads = tf.gradients(scaled_loss, all_vars)

    emb_grads = grads[:len(emb_vars)]
    emb_grads = [tf.IndexedSlices(grad.values * FLAGS.batch_size,
                                  grad.indices,
                                  grad.dense_shape) for grad in emb_grads]

    lstm_grads = grads[len(emb_vars):len(emb_vars) + len(lstm_vars)]
    lstm_grads, _ = tf.clip_by_global_norm(lstm_grads, FLAGS.max_grad_norm)

    softmax_grads = grads[len(emb_vars) + len(lstm_vars):]

    clipped_grads = emb_grads + lstm_grads + softmax_grads
    grads_and_vars = list(zip(clipped_grads, all_vars))

    optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate, initial_accumulator_value=1.0)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(decay=0.999)
    with tf.control_dependencies([train_op]):
      train_op = ema.apply(lstm_vars)

  model.global_step = global_step
  model.loss = loss
  model.train_op = train_op

  model.final_state_c = final_state_c
  model.final_state_h = final_state_h

  model.initial_state_c = initial_state_c
  model.initial_state_h = initial_state_h

  model.x = placeholder_x
  model.y = placeholder_y
  model.w = placeholder_w

  return model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, time, os, gc, math
from datetime import datetime

import numpy as np
import tensorflow as tf

import language_model as lm1b_model_graph
import lm1b_input

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 64, 'Batch size')
tf.flags.DEFINE_integer('num_steps', 20, 'Number of steps')
tf.flags.DEFINE_integer('num_eval_steps', 100, 'Number of eval steps')
tf.flags.DEFINE_string('data_dir', '/cmsdata/ssd1/cmslab/lm1b/', 'data directory')
tf.flags.DEFINE_string('model_dir', '/cmsdata/ssd1/cmslab/lm1b_graph_train', 'model directory(checkpoint)')
tf.flags.DEFINE_integer('evaluate_every_nth_ckpt', 1, 'Evaluate only every n-th checkpoint')

config = tf.ConfigProto(allow_soft_placement=True)

_NUM_WORDS = {
    'train': 798945280,
    'validation': 7789987,
}

_NUM_TRAIN_FILES = 99


def get_filenames(data_dir, is_training):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'training-monolingual.tokenized.shuffled', 'news.en-%05d-of-00100' % i)
        for i in range(1, _NUM_TRAIN_FILES + 1)]
  else:
    return [
        os.path.join(data_dir, 'heldout-monolingual.tokenized.shuffled', 'news.en-00000-of-00100')]

def input_data(data_dir, is_training):
  filenames = get_filenames(data_dir, is_training)
  vocab = lm1b_input.Vocabulary.from_file(os.path.join(data_dir, '1b_word_vocab.txt'))
  return lm1b_input.Dataset(vocab, filenames, deterministic=True)


def load_checkpoint(saver, sess, ckpt, nth_ckpt):
  if nth_ckpt >= len(ckpt.all_model_checkpoint_paths):
    raise RuntimeError('No more checkpoint file.')
  print("Evaluate checkpoint file [%d/%d]" % (nth_ckpt, len(ckpt.all_model_checkpoint_paths)))
  sys.stdout.flush()
  if ckpt and ckpt.all_model_checkpoint_paths[nth_ckpt]:
    if os.path.isabs(ckpt.all_model_checkpoint_paths[nth_ckpt]):
      model_checkpoint_path = ckpt.all_model_checkpoint_paths[nth_ckpt]
    else:
      raise RuntimeError('Checkpoint path should be absolute path.')
    global_step = ckpt.all_model_checkpoint_paths[nth_ckpt].split('/')[-1].split('-')[-1]
    if not global_step.isdigit():
      global_step = 0
    else:
      global_step = int(global_step)
    saver.restore(sess, model_checkpoint_path)
    print('Successfully loaded model from %s.' % model_checkpoint_path)
    sys.stdout.flush()
    return global_step
  else:
    raise RuntimeError('No checkpoint file found.')

def evaluate(sess, loss, final_state_c, final_state_h, placeholder_x, placeholder_y, placeholder_w, initial_state_c, initial_state_h, saver, ckpt, nth_ckpt):
  global_step = load_checkpoint(saver, sess, ckpt, nth_ckpt)
  sess.run(tf.local_variables_initializer())

  dataset = input_data(data_dir=FLAGS.data_dir, is_training=False)
  iterator = dataset.iterate_once(FLAGS.batch_size, FLAGS.num_steps)

  total_loss = 0.
  count = 0.
  try:
    state_c = np.zeros([int(d) for d in initial_state_c.shape.dims], dtype=np.float32)
    state_h = np.zeros([int(d) for d in initial_state_h.shape.dims], dtype=np.float32)
    for i, (x, y, w) in enumerate(iterator):
      if i >= FLAGS.num_eval_steps:
        break
      _loss, state_c, state_h = sess.run((loss, final_state_c, final_state_h), feed_dict={placeholder_x: x, placeholder_y: y, placeholder_w: w, initial_state_h: state_h, initial_state_c: state_c})
      total_loss += _loss
      count += w.mean()
      print("%d: %.3f (%.3f) ... " % (i, total_loss / count, np.exp(total_loss / count)))
      sys.stdout.flush()

  finally:
    log_perplexity = total_loss / count
    print('%s: global_step %d, log_perplexity = %.4f, perplexity = %.4f' % (datetime.now(), global_step, log_perplexity, np.exp(log_perplexity)))
    sys.stdout.flush()

new_names = {
  u'lm/B/ExponentialMovingAverage': u'model/model/lm/B/ExponentialMovingAverage',
  u'lm/W/ExponentialMovingAverage': u'model/model/lm/W/ExponentialMovingAverage',
  u'lm/W_P/ExponentialMovingAverage': u'model/model/lm/W_P/ExponentialMovingAverage',
  u'lm/softmax_b': u'model/lm/softmax_b',
}
for i in range(FLAGS.num_variable_shards):
  new_names[u'lm/emb_%d' % i] = u'model/lm/emb_%d' % i
  new_names[u'lm/softmax_w_%d' % i] = u'model/lm/softmax_w_%d' % i


def main():
  with tf.Graph().as_default() as g:
    with tf.device('/gpu:0'):
      model = lm1b_model_graph.LM(FLAGS.num_steps)
      placeholder_x = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.num_steps])
      placeholder_y = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.num_steps])
      placeholder_w = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.num_steps])
      initial_state_c = tf.placeholder(dtype=tf.float32,
                                       shape=[FLAGS.batch_size, model.state_size],
                                       name='initial_c')
      initial_state_h = tf.placeholder(dtype=tf.float32,
                                       shape=[FLAGS.batch_size, model.projected_size],
                                       name='initial_h')
      loss, final_state_c, final_state_h = model(placeholder_x, placeholder_y, placeholder_w, initial_state_c, initial_state_h, training=False)

    ema = tf.train.ExponentialMovingAverage(decay=0.999)
    lstm_vars = tf.trainable_variables()[-3:]
    avg_dict = ema.variables_to_restore(lstm_vars)
    new_dict = {}
    for key, value in avg_dict.items():
      new_dict[new_names[key]] = value
    saver = tf.train.Saver(new_dict)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    with tf.Session(config=config) as sess:
      for i in range(len(ckpt.all_model_checkpoint_paths)):
        if i % FLAGS.evaluate_every_nth_ckpt != 0:
          continue
        evaluate(sess, loss, final_state_c, final_state_h, placeholder_x, placeholder_y, placeholder_w, initial_state_c, initial_state_h, saver, ckpt, i)


if __name__ == "__main__":
  main()

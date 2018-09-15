import tensorflow as tf
import argparse
import os
import pickle
from model.auto_encoder import AutoEncoder
from model.language_model import LanguageModel
from data_utils import build_word_dict, build_word_dataset, batch_iter, download_dbpedia
import parallax_config
import parallax

NUM_EPOCHS = 10
MAX_DOCUMENT_LEN = 100


flags = tf.app.flags
flags.DEFINE_string('model', 'auto_encoder', '')
flags.DEFINE_string('data_dir', '', '')
flags.DEFINE_string('resource_info_file',
                    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 '.',
                                                 'resource_info')),
                    'Filename containing cluster information')
flags.DEFINE_integer('max_steps', 500000, '')
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_float('learning_rate', 0.001, '')
FLAGS = flags.FLAGS

def train():
    print("\nBuilding dictionary..")
    with open(os.path.join(FLAGS.data_dir,"word_dict.pickle"), "rb") as f:
        word_dict = pickle.load(f)
    single_gpu_graph = tf.Graph()
    with single_gpu_graph.as_default():
        if FLAGS.model == "auto_encoder":
            model = AutoEncoder(word_dict, MAX_DOCUMENT_LEN)
        elif FLAGS.model == "language_model":
            model = LanguageModel(word_dict, MAX_DOCUMENT_LEN)
        else:
            raise ValueError("Invalid model: {0}. Use auto_encoder | language_model".format(FLAGS.model))

        # Define training procedure
        global_step = tf.train.get_or_create_global_step()
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

    sess, num_workers, worker_id, num_replicas_per_worker = \
        parallax.parallel_run(single_gpu_graph,
                              FLAGS.resource_info_file,
                              parallax_config=parallax_config.build_config())

    def train_step(batch_x):
        feed_dict = {model.x: batch_x}
        _, step, loss = sess.run([train_op, global_step, model.loss], feed_dict=feed_dict)

        if step[0] % 100 == 0:
            print("step {0} : loss = {1}".format(step[0], loss[0]))

    print("Preprocessing dataset..")
    train_x, train_y = build_word_dataset("train", word_dict, MAX_DOCUMENT_LEN, data_dir=FLAGS.data_dir)

    # Training loop
    batch_x = []
    for i in range(num_replicas_per_worker):
        batches, _ = next(batch_iter(train_x, train_y, FLAGS.batch_size, NUM_EPOCHS, num_workers, worker_id))
        batch_x.append(batches)
    train_step(batch_x)

if __name__ == "__main__":
    train()

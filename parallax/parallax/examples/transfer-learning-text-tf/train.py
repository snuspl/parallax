import tensorflow as tf
import argparse
import os
from model.word_rnn import WordRNN
from data_utils import build_word_dict, build_word_dataset, batch_iter, download_dbpedia


NUM_CLASS = 14
BATCH_SIZE = 64
NUM_EPOCHS = 10
MAX_DOCUMENT_LEN = 100


def train(train_x, train_y, test_x, test_y, vocabulary_size, args):
    with tf.Session() as sess:
        model = WordRNN(vocabulary_size, MAX_DOCUMENT_LEN, NUM_CLASS)

        # Define training procedure
        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

        # Summary
        loss_summary = tf.summary.scalar("loss", model.loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Load variables from pre-trained model
        if not args.pre_trained == "none":
            pre_trained_variables = [v for v in tf.global_variables()
                                     if (v.name.startswith("embedding") or v.name.startswith("birnn")) and "Adam" not in v.name]
            saver = tf.train.Saver(pre_trained_variables)
            ckpt = tf.train.get_checkpoint_state(os.path.join(args.pre_trained, "model"))
            saver.restore(sess, ckpt.model_checkpoint_path)

        def train_step(batch_x, batch_y):
            feed_dict = {
                model.x: batch_x,
                model.y: batch_y,
                model.keep_prob: 0.5
            }

            _, step, summaries, loss = sess.run([train_op, global_step, summary_op, model.loss], feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)

            if step % 100 == 0:
                print("step {0} : loss = {1}".format(step, loss))

        def test_accuracy(test_x, test_y):
            test_batches = batch_iter(test_x, test_y, BATCH_SIZE, 1)
            sum_accuracy, cnt = 0, 0

            for test_batch_x, test_batch_y in test_batches:
                accuracy = sess.run(model.accuracy, feed_dict={model.x: test_batch_x, model.y: test_batch_y, model.keep_prob: 1.0})
                sum_accuracy += accuracy
                cnt += 1

            with open(args.summary_dir +"-accuracy.txt", "a") as f:
                print(sum_accuracy/cnt, file=f)

            return sum_accuracy / cnt

        # Training loop
        batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)

        for batch_x, batch_y in batches:
            train_step(batch_x, batch_y)
            step = tf.train.global_step(sess, global_step)

            if step % 200 == 0:
                test_acc = test_accuracy(test_x, test_y)
                print("test_accuracy = {0}\n".format(test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_trained", type=str, default="none", help="none | auto_encoder | language_model")
    parser.add_argument("--summary_dir", type=str, default="classifier", help="summary dir.")
    args = parser.parse_args()

    if not os.path.exists("dbpedia_csv"):
        print("Downloading dbpedia dataset...")
        download_dbpedia()

    print("\nBuilding dictionary..")
    word_dict = build_word_dict()
    print("Preprocessing dataset..")
    train_x, train_y = build_word_dataset("train", word_dict, MAX_DOCUMENT_LEN)
    test_x, test_y = build_word_dataset("test", word_dict, MAX_DOCUMENT_LEN)
    train(train_x, train_y, test_x, test_y, len(word_dict), args)

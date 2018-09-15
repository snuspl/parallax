import tensorflow as tf
from tensorflow.contrib import rnn


class WordRNN(object):
    def __init__(self, vocabulary_size, max_document_length, num_class):
        self.embedding_size = 256
        self.num_hidden = 512
        self.fc_num_hidden = 256

        self.x = tf.placeholder(tf.int32, [None, max_document_length])
        self.x_len = tf.reduce_sum(tf.sign(self.x), 1)
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32, [])

        with tf.variable_scope("embedding"):
            init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            x_emb = tf.nn.embedding_lookup(embeddings, self.x)

        with tf.variable_scope("rnn"):
            cell = rnn.BasicLSTMCell(self.num_hidden)
            rnn_outputs, _ = tf.nn.dynamic_rnn(
                cell, x_emb, sequence_length=self.x_len, dtype=tf.float32)
            rnn_output_flat = tf.reshape(rnn_outputs, [-1, max_document_length * self.num_hidden])

        with tf.name_scope("fc"):
            fc_output = tf.layers.dense(rnn_output_flat, self.fc_num_hidden, activation=tf.nn.relu)
            dropout = tf.nn.dropout(fc_output, self.keep_prob)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(dropout, num_class)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

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

import argparse
import os
import random
import numpy as np
import sys
import time

import nmt
import attention_model
import gnmt_model
import model as nmt_model
import parallax_config
import model_helper
from utils import misc_utils as utils
import train

FLAGS = None

def add_arguments(parser):
    """Build ArgumentParser for Parallax."""
    
    parser.add_argument("--resource_info_file", type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '.',
                                'resource_info')),
        help="Filename containing cluster information")
    parser.add_argument("--max_steps", type=int, default=1000000,
        help="Number of iterations to run for each workers.")
    parser.add_argument("--log_frequency", type=int, default=100,
        help="How many steps between two runop logs.")
    parser.add_argument("--sync", type="bool", nargs="?", const=True,
                        default=True)
    parser.add_argument('--epoch_size', type=int, default=0,
        help="total number of data instances")

def before_train(train_model, train_sess, global_step, hparams, log_f,
                 num_replicas_per_worker):
    """Misc tasks to do before training."""
    stats = train.init_stats()
    lr = train_sess.run(train_model.model.learning_rate)
    info = {"train_ppl": 0.0, "speed": 0.0, "avg_step_time": 0.0,
            "avg_grad_norm": 0.0,
            "learning_rate": lr}
    start_train_time = time.time()
    utils.print_out("# Start step %d, lr %g, %s" %
                   (global_step, info["learning_rate"], time.ctime()), log_f)

    # Initialize all of the iterators
    skip_count = hparams.batch_size * hparams.epoch_step
    utils.print_out("# Init train iterator, skipping %d elements" % skip_count)
    skip_count = train_model.skip_count_placeholder
    feed_dict = {}
    feed_dict[skip_count] = 0
    init = train_model.iterator.initializer
    train_sess.run(init, feed_dict=feed_dict)
    return stats, info, start_train_time

def main(_):
    default_hparams = nmt.create_hparams(FLAGS)
    ## Train / Decode
    out_dir = FLAGS.out_dir
    if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)

    # Load hparams.
    hparams = nmt.create_or_load_hparams(
        out_dir, default_hparams, FLAGS.hparams_path, save_hparams=False)

    log_device_placement = hparams.log_device_placement
    out_dir = hparams.out_dir
    num_train_steps = hparams.num_train_steps
    steps_per_stats = hparams.steps_per_stats
    avg_ckpts = hparams.avg_ckpts

    if not hparams.attention:
        model_creator = nmt_model.Model
    else:  # Attention
        if (hparams.encoder_type == "gnmt" or
            hparams.attention_architecture in ["gnmt", "gnmt_v2"]):
            model_creator = gnmt_model.GNMTModel
        elif hparams.attention_architecture == "standard":
            model_creator = attention_model.AttentionModel
        else:
            raise ValueError("Unknown attention architecture %s" %
                             hparams.attention_architecture)
    
    train_model =\
        model_helper.create_train_model(model_creator, hparams, scope=None, num_workers=1, jobid=0)

    config = utils.get_config_proto(
        log_device_placement=log_device_placement,
        num_intra_threads=1,
        num_inter_threads=36)

    def run(train_sess, num_workers, worker_id, num_replicas_per_worker, fetches):
         
        # Random
        random_seed = FLAGS.random_seed
        if random_seed is not None and random_seed > 0:
            utils.print_out("# Set random seed to %d" % random_seed)
            random.seed(random_seed + worker_id)
            np.random.seed(random_seed + worker_id)

        # Log and output files
        log_file = os.path.join(out_dir, "log_%d" % time.time())
        log_f = tf.gfile.GFile(log_file, mode="a")
        utils.print_out("# log_file=%s" % log_file, log_f)

        global_step = train_sess.run(train_model.model.global_step)
        last_stats_step = global_step

        # This is the training loop.
        stats, info, start_train_time = before_train(
            train_model, train_sess, global_step, hparams, log_f,
            num_replicas_per_worker)

        epoch_steps = FLAGS.epoch_size / (FLAGS.batch_size * num_workers * num_replicas_per_worker)

        for i in range(FLAGS.max_steps):
          ### Run a step ###
          start_time = time.time()
          if hparams.epoch_step !=0 and hparams.epoch_step % epoch_steps == 0:
              hparams.epoch_step  = 0
              skip_count = train_model.skip_count_placeholder
              feed_dict = {}
              feed_dict[skip_count] = [0 for i in range(num_replicas_per_worker)]
              init = train_model.iterator.initializer
              train_sess.run(init, feed_dict=feed_dict)

          _, sparse_grads_size_, sparse_grads_unique_size_ = train_sess.run(fetches)
          print('total_sparse_grads_size: %d, unique_sparse_grads_size: %d' % (np.sum(sparse_grads_size_), np.sum(sparse_grads_unique_size_)))

          hparams.epoch_step += 1

#    sess, num_workers, worker_id, num_replicas_per_worker = \
#        parallax.parallel_run(train_model.graph,
#                              FLAGS.resource_info_file,
#                              sync=FLAGS.sync,
#                              parallax_config=parallax_config.build_config())
    with train_model.graph.as_default():
        optimizer = train_model.model.optimizer
        params = train_model.model.params
        clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(
            train_model.model.gradients, max_gradient_norm=hparams.max_gradient_norm)
        train_model.model.grad_norm = grad_norm
        train_model.model.update = optimizer.apply_gradients(
            zip(clipped_grads, params), global_step=train_model.model.global_step)
        
        sparse_grads_size = []
        sparse_grads_unique_size= []
        for grad in train_model.model.gradients:
           if isinstance(grad, tf.IndexedSlices):
               sparse_grads_size = [tf.size(grad.indices, name='%s_size'%grad.op.name)]
               unique, _ = tf.unique(grad.indices)
               sparse_grads_unique_size = [tf.size(unique, name='%s_unique_size'%grad.op.name)]
        fetches = [[train_model.model.update.name], sparse_grads_size, sparse_grads_unique_size]
        sess = tf.train.MonitoredTrainingSession(config=config)
        run(sess, 1, 0, 1, fetches)

if __name__ == "__main__":
    import logging
    logging.getLogger("tensorflow").setLevel(logging.DEBUG)
    nmt_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmt_parser)
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

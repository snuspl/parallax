# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow benchmark library.

See the README for more information.
"""

import argparse
from collections import namedtuple
import math
import multiprocessing
import os
import threading
import time

from absl import flags
import numpy as np

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import nest

import convnet_builder
import datasets
from cnn_util import log_fn
from cnn_util import ParamSpec
from models import model_config
from platforms import util as platforms_util

# _DEFAULT_PARAMS maps from each parameter's name to its ParamSpec. For each
# parameter listed here, a command line flag will be defined for
# tf_cnn_benchmarks.py.
_DEFAULT_PARAMS = {
    'model':
        ParamSpec('string', 'trivial', 'name of the model to run'),

    # The code will first check if it's running under benchmarking mode
    # or evaluation mode, depending on 'eval':
    # Under the evaluation mode, this script will read a saved model,
    #   and compute the accuracy of the model against a validation dataset.
    #   Additional ops for accuracy and top_k predictors are only used under
    #   this mode.
    # Under the benchmarking mode, user can specify whether nor not to use
    #   the forward-only option, which will only compute the loss function.
    #   forward-only cannot be enabled with eval at the same time.
    'eval':
        ParamSpec('boolean', False, 'whether use eval or benchmarking'),
    'forward_only':
        ParamSpec('boolean', False,
                  'whether use forward-only or training for benchmarking'),
    'print_training_accuracy':
        ParamSpec('boolean', False,
                  'whether to calculate and print training accuracy during '
                  'training'),
    'batch_size':
        ParamSpec('integer', 0, 'batch size per compute device'),
    'batch_group_size':
        ParamSpec('integer', 10,
                  'number of groups of batches processed in the image '
                  'producer.'),
    'data_dir':
        ParamSpec('string', None,
                  'Path to dataset in TFRecord format (aka Example '
                  'protobufs). If not specified, synthetic data will be '
                  'used.'),
    'data_name':
        ParamSpec('string', None,
                  'Name of dataset: imagenet or cifar10. If not specified, it '
                  'is automatically guessed based on data_dir.'),
    'resize_method':
        ParamSpec('string', 'bilinear',
                  'Method for resizing input images: crop, nearest, bilinear, '
                  'bicubic, area, or round_robin. The `crop` mode requires '
                  'source images to be at least as large as the network input '
                  'size. The `round_robin` mode applies different resize '
                  'methods based on position in a batch in a round-robin '
                  'fashion. Other modes support any sizes and apply random '
                  'bbox distortions before resizing (even with '
                  'distortions=False).'),
    'distortions':
        ParamSpec('boolean', True,
                  'Enable/disable distortions during image preprocessing. '
                  'These include bbox and color distortions.'),
    'use_datasets':
        ParamSpec('boolean', True, 'Enable use of datasets for input pipeline'),
    'gpu_thread_mode':
        ParamSpec('string', 'gpu_private',
                  'Methods to assign GPU host work to threads. '
                  'global: all GPUs and CPUs share the same global threads; '
                  'gpu_private: a private threadpool for each GPU; '
                  'gpu_shared: all GPUs share the same threadpool.'),
    'per_gpu_thread_count':
        ParamSpec('integer', 0, 'The number of threads to use for GPU.'
                                'Only valid when gpu_thread_mode is not global.'),
    'cache_data':
        ParamSpec('boolean', False,
                  'Enable use of a special datasets pipeline that reads a '
                  'single TFRecord into memory and repeats it infinitely many '
                  'times. The purpose of this flag is to make it possible '
                  'to write regression tests that are not bottlenecked by CNS '
                  'throughput.'),
    'data_format':
        ParamSpec('string', 'NCHW',
                  'Data layout to use: NHWC (TF native) or NCHW (cuDNN '
                  'native, requires GPU).'),
    'num_intra_threads':
        ParamSpec('integer', 1,
                  'Number of threads to use for intra-op parallelism. If set '
                  'to 0, the system will pick an appropriate number.'),
    'num_inter_threads':
        ParamSpec('integer', 0,
                  'Number of threads to use for inter-op parallelism. If set '
                  'to 0, the system will pick an appropriate number.'),
    'params_stat':
        ParamSpec('boolean', 'False',
                  'Enable params stat info'),
    'optimizer':
        ParamSpec('string', 'sgd',
                  'Optimizer to use: momentum or sgd or rmsprop'),
    'num_total_devices':
        ParamSpec('integer', 0,
                  'total number of GPU devices'),
    'learning_rate':
        ParamSpec('float', None, 'Initial learning rate for training.'),
    'num_epochs_per_decay':
        ParamSpec('float', 0,
                  'Steps after which learning rate decays. If 0, the learning '
                  'rate does not decay.'),
    'learning_rate_decay_factor':
        ParamSpec('float', 0,
                  'Learning rate decay factor. Decay by this factor every '
                  '`num_epochs_per_decay` epochs. If 0, learning rate does '
                  'not decay.'),
    'minimum_learning_rate':
        ParamSpec('float', 0,
                  'The minimum learning rate. The learning rate will '
                  'never decay past this value. Requires `learning_rate`, '
                  '`num_epochs_per_decay` and `learning_rate_decay_factor` to '
                  'be set.'),
    'momentum':
        ParamSpec('float', 0.9, 'Momentum for training.'),
    'rmsprop_decay':
        ParamSpec('float', 0.9, 'Decay term for RMSProp.'),
    'rmsprop_momentum':
        ParamSpec('float', 0.9, 'Momentum in RMSProp.'),
    'rmsprop_epsilon':
        ParamSpec('float', 1.0, 'Epsilon term for RMSProp.'),
    'gradient_clip':
        ParamSpec('float', None,
                  'Gradient clipping magnitude. Disabled by default.'),
    'weight_decay':
        ParamSpec('float', 0.00004, 'Weight decay factor for training.'),
    'gpu_memory_frac_for_testing':
        ParamSpec('float', 0,
                  'If non-zero, the fraction of GPU memory that will be used. '
                  'Useful for testing the benchmark script, as this allows '
                  'distributed mode to be run on a single machine. For '
                  'example, if there are two tasks, each can be allocated '
                  '~40 percent of the memory on a single machine'),
    'use_tf_layers':
        ParamSpec('boolean', True,
                  'If True, use tf.layers for neural network layers. This '
                  'should not affect performance or accuracy in any way.'),
    'tf_random_seed':
        ParamSpec('integer', 1234,
                  'The TensorFlow random seed. Useful for debugging NaNs, as '
                  'this can be set to various values to see if the NaNs '
                  'depend on the seed.'),

    # Performance tuning parameters.
    'winograd_nonfused':
        ParamSpec('boolean', True,
                  'Enable/disable using the Winograd non-fused algorithms.'),
    'sync_on_finish':
        ParamSpec('boolean', False,
                  'Enable/disable whether the devices are synced after each '
                  'step.'),
    'force_gpu_compatible':
        ParamSpec('boolean', True,
                  'whether to enable force_gpu_compatible in GPU_Options'),
    'fuse_decode_and_crop':
        ParamSpec('boolean', True,
                  'Fuse decode_and_crop for image preprocessing.'),
    'distort_color_in_yiq':
        ParamSpec('boolean', True,
                  'Distort color of input images in YIQ space.'),
    # Performance tuning specific to MKL.
    'mkl':
        ParamSpec('boolean', False, 'If true, set MKL environment variables.'),
    'kmp_blocktime':
        ParamSpec('integer', 30,
                  'The time, in milliseconds, that a thread should wait, '
                  'after completing the execution of a parallel region, '
                  'before sleeping'),
    'kmp_affinity':
        ParamSpec('string', 'granularity=fine,verbose,compact,1,0',
                  'Restricts execution of certain threads (virtual execution '
                  'units) to a subset of the physical processing units in a '
                  'multiprocessor computer.'),
    'kmp_settings':
        ParamSpec('integer', 1, 'If set to 1, MKL settings will be printed.'),
    # fp16 parameters. If use_fp16=False, no other fp16 parameters apply.
    'use_fp16':
        ParamSpec('boolean', False,
                  'Use 16-bit floats for certain tensors instead of 32-bit '
                  'floats. This is currently experimental.'),
    # TODO(reedwm): The default loss scale of 128 causes most models to diverge
    # on the second step with synthetic data. Changing the tf.set_random_seed
    # call to tf.set_random_seed(1235) or most other seed values causes the
    # issue not to occur.
    'fp16_loss_scale':
        ParamSpec('float', None,
                  'If fp16 is enabled, the loss is multiplied by this amount '
                  'right before gradients are computed, then each gradient '
                  'is divided by this amount. Mathematically, this has no '
                  'effect, but it helps avoid fp16 underflow. Set to 1 to '
                  'effectively disable.'),
    # Evaluation parameters
    'eval_interval_secs':
        ParamSpec('integer', 0, 'If set to 0, evaluate only once.'),
    'checkpoint_dir':
        ParamSpec('string', None, 'Log directory which is used for evaluation'),
    'display_every_for_eval':
        ParamSpec('integer', 10,
                  'number of steps between two consecutive logging'
                  'for evaluation.'),
    # TODO: the number of batches for evaluation should be automatically decided
    #       and computation for this value is the following:
    #       num_batches_for_eval = test_dataset_size / batch_size
    'num_batches_for_eval':
        ParamSpec('integer', 20, 'number of batches for evaluation'),

    'deterministic':
        ParamSpec('boolean', False, 'Use deterministic model or not'),
}
_DEFAULT_PARAMS.update(platforms_util.get_platform_params())


def define_flags():
    """Define a command line flag for each ParamSpec in _DEFAULT_PARAMS."""
    define_flag = {
        'boolean': flags.DEFINE_boolean,
        'float': flags.DEFINE_float,
        'integer': flags.DEFINE_integer,
        'string': flags.DEFINE_string,
    }
    for name, param_spec in six.iteritems(_DEFAULT_PARAMS):
        if param_spec.flag_type not in define_flag:
            raise ValueError('Unknown flag_type %s' % param_spec.flag_type)
        else:
            define_flag[param_spec.flag_type](name, param_spec.default_value,
                                              param_spec.description)
            flags.declare_key_flag(name)


FLAGS = flags.FLAGS


class CheckpointNotFoundException(Exception):
    pass


def get_data_type(params):
    """Returns BenchmarkCNN's data type as determined by use_fp16.

    Args:
      params: Params tuple, typically created by make_params or
            make_params_from_flags.
    """
    return tf.float16 if params.use_fp16 else tf.float32


def loss_function(logits, labels, aux_logits):
    """Loss function."""
    with tf.name_scope('xentropy'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    if aux_logits is not None:
        with tf.name_scope('aux_xentropy'):
            aux_cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                logits=aux_logits, labels=labels)
            aux_loss = 0.4 * tf.reduce_mean(aux_cross_entropy, name='aux_loss')
            loss = tf.add_n([loss, aux_loss])
    return loss


def create_config_proto(params):
    """Returns session config proto.

    Args:
      params: Params tuple, typically created by make_params or
            make_params_from_flags.
    """
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = params.num_intra_threads
    config.inter_op_parallelism_threads = params.num_inter_threads
    config.gpu_options.force_gpu_compatible = params.force_gpu_compatible
    if params.gpu_memory_frac_for_testing > 0:
        config.gpu_options.per_process_gpu_memory_fraction = (
            params.gpu_memory_frac_for_testing)
    return config


def get_mode_from_params(params):
    """Returns the mode in which this script is running.

    Args:
      params: Params tuple, typically created by make_params or
            make_params_from_flags.
    Raises:
      ValueError: Unsupported params settings.
    """
    if params.forward_only and params.eval:
        raise ValueError('Only one of forward_only and eval parameters is true')

    if params.eval:
        return 'evaluation'
    if params.forward_only:
        return 'forward-only'
    return 'training'


# Params are passed to BenchmarkCNN's constructor. Params is a map from name
# to value, with one field per key in _DEFAULT_PARAMS.
#
# Call make_params() or make_params_from_flags() below to construct a Params
# tuple with default values from _DEFAULT_PARAMS, rather than constructing
# Params directly.
Params = namedtuple('Params',
                    _DEFAULT_PARAMS.keys())  # pylint: disable=invalid-name


def make_params(**kwargs):
    """Create a Params tuple for BenchmarkCNN from kwargs.

    Default values are filled in from _DEFAULT_PARAMS.

    Args:
      **kwargs: kwarg values will override the default values.
    Returns:
      Params namedtuple for constructing BenchmarkCNN.
    """
    # Create a (name: default_value) map from PARAMS.
    default_kwargs = {
        name: _DEFAULT_PARAMS[name].default_value
        for name in _DEFAULT_PARAMS
    }
    return Params(**default_kwargs)._replace(**kwargs)


def load_checkpoint(saver, sess, checkpoint_dir, nth_ckpt):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if nth_ckpt >= len(ckpt.all_model_checkpoint_paths):
        raise CheckpointNotFoundException('No more checkpoint file.')
    log_fn("Evaluate checkpoint file [%d/%d]" % (
    nth_ckpt, len(ckpt.all_model_checkpoint_paths)))
    if ckpt and ckpt.all_model_checkpoint_paths[nth_ckpt]:
        if os.path.isabs(ckpt.all_model_checkpoint_paths[nth_ckpt]):
            model_checkpoint_path = ckpt.all_model_checkpoint_paths[nth_ckpt]
        else:
            raise ValueError('Checkpoint path should be absolute path.')
        global_step = \
        ckpt.all_model_checkpoint_paths[nth_ckpt].split('/')[-1].split('-')[-1]
        if not global_step.isdigit():
            global_step = 0
        else:
            global_step = int(global_step)
        saver.restore(sess, model_checkpoint_path)
        log_fn('Successfully loaded model from %s.' % model_checkpoint_path)
        return global_step
    else:
        raise CheckpointNotFoundException('No checkpoint file found.')


def make_params_from_flags():
    """Create a Params tuple for BenchmarkCNN from FLAGS.

    Returns:
      Params namedtuple for constructing BenchmarkCNN.
    """
    # Collect (name: value) pairs for FLAGS with matching names in
    # _DEFAULT_PARAMS.
    flag_values = {name: getattr(FLAGS, name) for name in
                   _DEFAULT_PARAMS.keys()}
    return Params(**flag_values)


def get_piecewise_learning_rate(piecewise_learning_rate_schedule,
                                global_step, num_batches_per_epoch):
    """Returns a piecewise learning rate tensor.

    Args:
      piecewise_learning_rate_schedule: The --piecewise_learning_rate_schedule
        parameter
      global_step: Scalar tensor representing the global step.
      num_batches_per_epoch: float indicating the number of batches per epoch.

    Returns:
      A scalar float tensor, representing the learning rate.

    Raises:
      ValueError: piecewise_learning_rate_schedule is not formatted correctly.
    """
    pieces = piecewise_learning_rate_schedule.split(';')
    if len(pieces) % 2 == 0:
        raise ValueError('--piecewise_learning_rate_schedule must have an odd '
                         'number of components')
    values = []
    boundaries = []
    for i, piece in enumerate(pieces):
        if i % 2 == 0:
            try:
                values.append(float(piece))
            except ValueError:
                raise ValueError('Invalid learning rate: ' + piece)
        else:
            try:
                boundaries.append(int(int(piece) * num_batches_per_epoch) - 1)
            except ValueError:
                raise ValueError('Invalid epoch: ' + piece)
    return tf.train.piecewise_constant(global_step, boundaries, values,
                                       name='piecewise_learning_rate')


def get_learning_rate(params, global_step, num_examples_per_epoch, model,
                      batch_size):
    """Returns a learning rate tensor based on global_step.

    Args:
      params: Params tuple, typically created by make_params or
        make_params_from_flags.
      global_step: Scalar tensor representing the global step.
      num_examples_per_epoch: The number of examples per epoch.
      model: The model.Model object to obtain the default learning rate from if no
        learning rate is specified.
      batch_size: Number of examples per step

    Returns:
      A scalar float tensor, representing the learning rate. When evaluated, the
      learning rate depends on the current value of global_step.

    Raises:
      ValueError: Invalid or unsupported params.
    """
    num_batches_per_epoch = (float(num_examples_per_epoch) / batch_size)

    learning_rate = (
            params.learning_rate or model.get_learning_rate(global_step,
                                                            batch_size))
    if (params.learning_rate and params.num_epochs_per_decay > 0 and
            params.learning_rate_decay_factor > 0):
        decay_steps = int(num_batches_per_epoch * params.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        learning_rate = tf.train.exponential_decay(
            params.learning_rate,
            global_step,
            decay_steps,
            params.learning_rate_decay_factor,
            staircase=True)

        if params.minimum_learning_rate != 0.:
            learning_rate = tf.maximum(learning_rate,
                                       params.minimum_learning_rate)
    return learning_rate


class BenchmarkCNN(object):
    """Class for benchmarking a cnn network."""

    def __init__(self, params):
        """Initialize BenchmarkCNN.

        Args:
          params: Params tuple, typically created by make_params or
                  make_params_from_flags.
        Raises:
          ValueError: Unsupported params settings.
        """
        self.params = params
        if FLAGS.deterministic:
            assert self.params.data_dir is None
            self.dataset = datasets.create_dataset(None,
                                                    self.params.data_name)
        else:
            self.dataset = datasets.create_dataset(self.params.data_dir,
                                               self.params.data_name)
        self.model = model_config.get_model_config(self.params.model,
                                                   self.dataset)
        self.data_format = self.params.data_format
        self.resize_method = self.params.resize_method
        self.use_synthetic_gpu_images = self.dataset.use_synthetic_gpu_images()
        self.num_batches_for_eval = self.params.num_batches_for_eval

        if ((self.params.num_epochs_per_decay or
             self.params.learning_rate_decay_factor) and
                not (
                        self.params.learning_rate and self.params.num_epochs_per_decay and
                        self.params.learning_rate_decay_factor)):
            raise ValueError('If one of num_epochs_per_decay or '
                             'learning_rate_decay_factor is set, both must be set'
                             'and learning_rate must be set')
        if (self.params.minimum_learning_rate and
                not (
                        self.params.learning_rate and self.params.num_epochs_per_decay and
                        self.params.learning_rate_decay_factor)):
            raise ValueError('minimum_learning_rate requires learning_rate,'
                             'num_epochs_per_decay, and '
                             'learning_rate_decay_factor to be set')

        # Use the batch size from the command line if specified, otherwise use the
        # model's default batch size.  Scale the benchmark's batch size by the
        # number of GPUs.
        if self.params.batch_size > 0:
            self.model.set_batch_size(self.params.batch_size)
        self.batch_size = self.model.get_batch_size()
        self.batch_group_size = self.params.batch_group_size
        self.loss_scale = None
        self.loss_scale_normal_steps = None
        self.image_preprocessor = self.get_image_preprocessor()

    def print_info(self):
        """Print basic information."""
        log_fn('Model:       %s' % self.model.get_model())
        dataset_name = self.dataset.name
        if self.dataset.use_synthetic_gpu_images():
            dataset_name += ' (synthetic)'
        log_fn('Dataset:     %s' % dataset_name)
        log_fn('Mode:        %s' % get_mode_from_params(self.params))
        log_fn('Batch size:  %s per device' % self.batch_size)
        if self.batch_group_size > 1:
            log_fn('             %d batches per prepocessing group' %
                   self.batch_group_size)
        log_fn('Data format: %s' % self.data_format)
        log_fn('Optimizer:   %s' % self.params.optimizer)
        log_fn('==========')

    def build_model(self):
        """Run the benchmark task assigned to this process.

        Returns:
          Dictionary of statistics for training or eval.
        Raises:
           ValueError: unrecognized job name.
        """
        if self.params.eval:
            raise ValueError('Use evaluate() function instead.')
        else:
            self._benchmark_cnn()

    def evaluate(self):
        """Evaluate model in saved checkpoints."""
        (image_producer_ops, fetches) = self._build_model()

        saver = tf.train.Saver(tf.global_variables())
        local_var_init_op = tf.local_variables_initializer()
        variable_mgr_init_ops = [local_var_init_op]
        local_var_init_op_group = tf.group(*variable_mgr_init_ops)

        nth_ckpt = 33
        while True:
            self._eval_once(saver, image_producer_ops, fetches,
                            local_var_init_op_group, nth_ckpt)
            nth_ckpt += 1
            if self.params.eval_interval_secs <= 0:
                break
            time.sleep(self.params.eval_interval_secs)

    def _eval_once(self, saver, image_producer_ops, fetches,
                   local_var_init_op_group, nth_ckpt):
        with tf.Session(config=create_config_proto(self.params)) as sess:

            coord = tf.train.Coordinator()
            if self.params.checkpoint_dir is None:
                raise ValueError(
                    'Checkpoint directory for evaluation is not specified')
            try:
                global_step = load_checkpoint(saver, sess,
                                              self.params.checkpoint_dir,
                                              nth_ckpt)
            except CheckpointNotFoundException:
                log_fn(
                    'Checkpoint not found in %s' % self.params.checkpoint_dir)
                sys.exit(-1)
                return
            log_fn('[Evaluation] START')
            sess.run(local_var_init_op_group)

            assert not self.use_synthetic_gpu_images

            dummy_queue = tf.FIFOQueue(1, [tf.bool], shapes=[[]],
                                       name='dummy_queue',
                                       shared_name='dummy_queue')

            qr = tf.train.QueueRunner(dummy_queue, image_producer_ops)
            tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
            enqueue_threads = qr.create_threads(sess=sess, coord=coord,
                                                daemon=True)
            for thread in enqueue_threads:
                thread.start()

            top_1_accuracy_sum = 0.0
            top_5_accuracy_sum = 0.0
            total_eval_count = self.num_batches_for_eval * self.batch_size
            for step in xrange(self.num_batches_for_eval):
                results = sess.run(fetches)
                top_1_accuracy_sum += results['top_1_accuracy']
                top_5_accuracy_sum += results['top_5_accuracy']
                if (step + 1) % self.params.display_every_for_eval == 0:
                    log_fn('%i\ttop_1_accuracy: %.4f' % (
                    step + 1, top_1_accuracy_sum / step))
                    log_fn('%i\ttop_5_accuracy: %.4f' % (
                    step + 1, top_5_accuracy_sum / step))
            accuracy_at_1 = top_1_accuracy_sum / self.num_batches_for_eval
            accuracy_at_5 = top_5_accuracy_sum / self.num_batches_for_eval
            log_fn(
                '[SUMMARY] Global step: %d Accuracy @ 1 = %.4f Accuracy @ 5 = %.4f [%d examples]' %
                (global_step, accuracy_at_1, accuracy_at_5, total_eval_count))
            sess.run(dummy_queue.close(cancel_pending_enqueues=True))
            coord.request_stop()

    def _benchmark_cnn(self):
        """Run cnn in benchmark mode. When forward_only on, it forwards CNN.

        Returns:
          Dictionary containing training statistics (num_workers, num_steps,
          average_wall_time, images_per_sec).
        """
        (image_producer_ops, fetches) = self._build_model()
        fetches_list = nest.flatten(list(fetches.values()))
        main_fetch_group = tf.group(*fetches_list)
        global_step = tf.train.get_global_step()

        with tf.device('/cpu:0'):
            with tf.control_dependencies([main_fetch_group]):
                self.train_op = global_step.assign_add(1, use_locking=True)

        local_var_init_op = tf.local_variables_initializer()
        variable_mgr_init_ops = [local_var_init_op]
        local_var_init_op_group = tf.group(*variable_mgr_init_ops)

        if not self.use_synthetic_gpu_images:
            dummy_queue = tf.FIFOQueue(1, [tf.bool], shapes=[[]],
                                       name='dummy_queue',
                                       shared_name='dummy_queue')
            qr = tf.train.QueueRunner(dummy_queue, image_producer_ops)
            tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)

    def _build_image_processing(self, shift_ratio=0):
        """"Build the image (pre)processing portion of the model graph."""
        if self.use_synthetic_gpu_images:
            return (None, None)

        with tf.device('/cpu:0'):
            if self.params.eval:
                subset = 'validation'
            else:
                subset = 'train'
            image_producer_ops = []
            images_splits, labels_splits = self.image_preprocessor.minibatch(
                self.dataset,
                subset=subset,
                use_datasets=self.params.use_datasets,
                cache_data=self.params.cache_data,
                shift_ratio=shift_ratio)
            images_shape = images_splits[0].get_shape()
            labels_shape = labels_splits[0].get_shape()

        with tf.device('/gpu:0'):
            if self.params.eval:
                image_producer_stage = data_flow_ops.StagingArea(
                    [images_splits[0].dtype, labels_splits[0].dtype],
                    shapes=[images_shape, labels_shape],
                    capacity=1)
            else:
                image_producer_stage = data_flow_ops.StagingArea(
                    [images_splits[0].dtype, labels_splits[0].dtype],
                    shapes=[images_shape, labels_shape],
                    capacity=self.batch_group_size)

            put_op = image_producer_stage.put(
                [images_splits[0], labels_splits[0]])
            image_producer_ops.append(put_op)
        return (image_producer_ops, image_producer_stage)

    def _build_model(self):
        """Build the TensorFlow graph."""
        tf.set_random_seed(self.params.tf_random_seed)
        np.random.seed(4321)
        phase_train = not (self.params.eval or self.params.forward_only)

        losses = []
        device_grads = []
        all_logits = []
        all_top_1_ops = []
        all_top_5_ops = []

        with tf.device('/cpu:0'):
            global_step = tf.train.get_or_create_global_step()
            self.global_step = global_step
            if self.params.use_fp16:
                init_loss_scale_val = float(self.params.fp16_loss_scale or
                                            self.model.get_fp16_loss_scale())
                if init_loss_scale_val != 1:
                    self.loss_scale = tf.get_variable(
                        name='loss_scale',
                        initializer=init_loss_scale_val,
                        dtype=tf.float32,
                        trainable=False)

        # Build the processing and model for the worker.
        (image_producer_ops,
         image_producer_stage) = self._build_image_processing(shift_ratio=0)
        staging_delta_ops = []

        results = self.add_forward_pass_and_gradients(
            phase_train, image_producer_stage)
        if phase_train:
            losses.append(results['loss'])
            device_grads.append(results['gradvars'])
        else:
            all_logits.append(results['logits'])
        if not phase_train or self.params.print_training_accuracy:
            all_top_1_ops.append(results['top_1_op'])
            all_top_5_ops.append(results['top_5_op'])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        fetches = self._build_fetches(global_step, all_logits, losses,
                                      device_grads,
                                      update_ops, all_top_1_ops, all_top_5_ops,
                                      phase_train)
        return (image_producer_ops, fetches)

    def _build_fetches(self, global_step, all_logits, losses, device_grads,
                       update_ops, all_top_1_ops, all_top_5_ops, phase_train):
        """Complete construction of model graph, populating the fetches map."""
        fetches = {}
        if all_top_1_ops:
            fetches['top_1_accuracy'] = tf.reduce_sum(
                all_top_1_ops) / self.batch_size
        if all_top_5_ops:
            fetches['top_5_accuracy'] = tf.reduce_sum(
                all_top_5_ops) / self.batch_size

        if not phase_train:
            if self.params.forward_only:
                fetches['all_logits'] = tf.concat(all_logits, 0)
            return fetches

        training_ops = []
        with tf.device('/gpu:0'):
            total_loss = tf.reduce_mean(losses)
            self.cost = total_loss
            avg_grads = device_grads[0]

            gradient_clip = self.params.gradient_clip
            if 'resnet' in self.params.model:
                learning_rate = self.model.get_learning_rate(global_step,
                                                             self.params.num_total_devices * self.batch_size)
            else:
                learning_rate = get_learning_rate(self.params, global_step,
                                                  self.dataset.num_examples_per_epoch(),
                                                  self.model, self.batch_size)
            self.lr = learning_rate
            gradients_size = []
            params_size = []
            for grad, var in avg_grads:
                if isinstance(grad, tf.IndexedSlices):
                    gradients_size.append(tf.size(grad.values))
                    gradients_size.append(tf.size(grad.indices))
                else:
                    gradients_size.append(tf.size(grad))
                params_size.append(tf.size(var))
            if self.params.params_stat:
                total_gradients_size = tf.add_n(gradients_size)
                total_params_size = tf.add_n(params_size)
            if gradient_clip is not None:
                clipped_grads = [(tf.clip_by_value(grad, -gradient_clip,
                                                   +gradient_clip), var)
                                 for grad, var in avg_grads]
            else:
                clipped_grads = avg_grads

            learning_rate = tf.identity(learning_rate, name='learning_rate')
            if self.params.optimizer == 'momentum':
                opt = tf.train.MomentumOptimizer(
                    learning_rate, self.params.momentum, use_nesterov=True)
            elif self.params.optimizer == 'sgd':
                opt = tf.train.GradientDescentOptimizer(learning_rate)
            elif self.params.optimizer == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(
                    learning_rate,
                    self.params.rmsprop_decay,
                    momentum=self.params.rmsprop_momentum,
                    epsilon=self.params.rmsprop_epsilon)
            else:
                raise ValueError('Optimizer "%s" was not recognized',
                                 self.params.optimizer)

            training_ops.extend([opt.apply_gradients(clipped_grads)])
        train_ops = tf.group(*(training_ops + update_ops))

        fetches['train_op'] = train_ops
        fetches['total_loss'] = total_loss
        if self.params.params_stat:
            fetches['total_gradients_size'] = total_gradients_size
            fetches['total_params_size'] = total_params_size
        return fetches

    def add_forward_pass_and_gradients(self, phase_train, image_producer_stage):
        """Add ops for forward-pass and gradient computations."""
        nclass = self.dataset.num_classes + 1
        input_data_type = get_data_type(self.params)
        data_type = get_data_type(self.params)
        with tf.device('/gpu:0'):
            if not self.use_synthetic_gpu_images:
                images, labels = image_producer_stage.get()
            else:
                # Minor hack to avoid H2D copy when using synthetic data
                image_size = self.model.get_image_size()
                image_shape = [
                    self.batch_size, image_size, image_size,
                    self.dataset.depth
                ]
                labels_shape = [self.batch_size]
                # Synthetic image should be within [0, 255].
                images = tf.truncated_normal(
                    image_shape,
                    dtype=input_data_type,
                    mean=127,
                    stddev=60,
                    name='synthetic_images')
                images = tf.contrib.framework.local_variable(
                    images, name='gpu_cached_images')
                labels = tf.random_uniform(
                    labels_shape,
                    minval=0,
                    maxval=nclass - 1,
                    dtype=tf.int32,
                    name='synthetic_labels')

        # Rescale from [0, 255] to [0, 2]
        images = tf.multiply(images, 1. / 127.5)
        # Rescale to [-1, 1]
        images = tf.subtract(images, 1.0)

        if self.data_format == 'NCHW':
            images = tf.transpose(images, [0, 3, 1, 2])
        if input_data_type != data_type:
            images = tf.cast(images, data_type)
        var_type = tf.float32
        network = convnet_builder.ConvNetBuilder(
            images, self.dataset.depth, phase_train, self.params.use_tf_layers,
            self.data_format, data_type, var_type)
        with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
            self.model.add_inference(network)
            # Add the final fully-connected class layer
            logits = network.affine(nclass, activation='linear')
            aux_logits = None
            if network.aux_top_layer is not None:
                with network.switch_to_aux_top_layer():
                    aux_logits = network.affine(
                        nclass, activation='linear', stddev=0.001)
        if data_type == tf.float16:
            # TODO(reedwm): Determine if we should do this cast here.
            logits = tf.cast(logits, tf.float32)
            if aux_logits is not None:
                aux_logits = tf.cast(aux_logits, tf.float32)

        results = {}  # The return value
        if not phase_train or self.params.print_training_accuracy:
            top_1_op = tf.reduce_sum(
                tf.cast(tf.nn.in_top_k(logits, labels, 1), data_type))
            top_5_op = tf.reduce_sum(
                tf.cast(tf.nn.in_top_k(logits, labels, 5), data_type))
            results['top_1_op'] = top_1_op
            results['top_5_op'] = top_5_op

        if not phase_train:
            results['logits'] = logits
            return results
        loss = loss_function(logits, labels, aux_logits=aux_logits)
        params = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params])
        weight_decay = self.params.weight_decay
        if weight_decay is not None and weight_decay != 0.:
            loss += weight_decay * l2_loss

        aggmeth = tf.AggregationMethod.DEFAULT
        scaled_loss = loss if self.loss_scale is None else loss * self.loss_scale
        grads = tf.gradients(scaled_loss, params, aggregation_method=aggmeth)
        if self.loss_scale is not None:
            # TODO(reedwm): If automatic loss scaling is not used, we could avoid
            # these multiplications by directly modifying the learning rate instead.
            # If this is done, care must be taken to ensure that this scaling method
            # is correct, as some optimizers square gradients and do other
            # operations which might not be compatible with modifying both the
            # gradients and the learning rate.
            grads = [
                grad * tf.cast(1. / self.loss_scale, grad.dtype) for grad in
                grads
            ]
        param_refs = tf.trainable_variables()
        gradvars = list(zip(grads, param_refs))
        results['loss'] = loss
        results['gradvars'] = gradvars
        return results

    def get_image_preprocessor(self):
        """Returns the image preprocessor to used, based on the model.

        Returns:
          The image preprocessor, or None if synthetic data should be used.
        """
        image_size = self.model.get_image_size()
        input_data_type = get_data_type(self.params)

        shift_ratio = 0

        processor_class = self.dataset.get_image_preprocessor()
        assert processor_class
        return processor_class(
            image_size,
            image_size,
            self.batch_size,
            1,
            dtype=input_data_type,
            train=(not self.params.eval),
            distortions=self.params.distortions,
            resize_method=self.resize_method,
            shift_ratio=shift_ratio,
            summary_verbosity=0,
            distort_color_in_yiq=self.params.distort_color_in_yiq,
            fuse_decode_and_crop=self.params.fuse_decode_and_crop)


def setup(params):
    """Sets up the environment that BenchmarkCNN should run in.

    Args:
      params: Params tuple, typically created by make_params or
            make_params_from_flags.
    Returns:
      A potentially modified params.
    Raises:
      ValueError: invalid parames combinations.
    """
    if params.winograd_nonfused:
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    else:
        os.environ.pop('TF_ENABLE_WINOGRAD_NONFUSED', None)
    os.environ['TF_SYNC_ON_FINISH'] = str(int(params.sync_on_finish))
    argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Sets environment variables for MKL
    if params.mkl:
        os.environ['KMP_BLOCKTIME'] = str(params.kmp_blocktime)
        os.environ['KMP_SETTINGS'] = str(params.kmp_settings)
        os.environ['KMP_AFFINITY'] = params.kmp_affinity
        if params.num_intra_threads > 0:
            os.environ['OMP_NUM_THREADS'] = str(params.num_intra_threads)

    # Sets GPU thread settings
    params = params._replace(gpu_thread_mode=params.gpu_thread_mode.lower())
    if params.gpu_thread_mode not in ['global', 'gpu_shared', 'gpu_private']:
        raise ValueError('Invalid gpu_thread_mode: %s' % params.gpu_thread_mode)
    os.environ['TF_GPU_THREAD_MODE'] = params.gpu_thread_mode

    if params.per_gpu_thread_count and params.gpu_thread_mode == 'global':
        raise ValueError(
            'Invalid per_gpu_thread_count with gpu_thread_mode=global: %s' %
            params.per_gpu_thread_count)
    # Default to two threads. One for the device compute and the other for
    # memory copies.
    per_gpu_thread_count = params.per_gpu_thread_count or 2
    total_gpu_thread_count = per_gpu_thread_count

    if params.gpu_thread_mode == 'gpu_private':
        os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
    elif params.gpu_thread_mode == 'gpu_shared':
        os.environ['TF_GPU_THREAD_COUNT'] = str(total_gpu_thread_count)

    if not params.num_inter_threads and params.gpu_thread_mode in [
        'gpu_private', 'gpu_shared'
    ]:
        cpu_count = multiprocessing.cpu_count()
        main_thread_count = max(cpu_count - total_gpu_thread_count, 1)
        params = params._replace(num_inter_threads=main_thread_count)

    config = create_config_proto(params)
    platforms_util.initialize(params, create_config_proto(params))

    return params, config

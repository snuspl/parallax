
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


flags = tf.app.flags
flags.DEFINE_boolean('replicate_variables', True, """replicate_variables""")
flags.DEFINE_string('protocol', 'grpc', """The method for managing variables""")
tf.app.flags.DEFINE_string('mpirun_options', '', 'option for mpirun')
flags.DEFINE_string('run_option', 'HYBRID',
                    'The run option whether PS, MPI or HYBRID')
flags.DEFINE_string('redirect_path', None, """redirect path to keep the log of distributed workers""")
flags.DEFINE_integer('save_ckpt_steps', None,
                     """Number of steps between two consecutive checkpoints""")
flags.DEFINE_integer('save_n_ckpts_per_epoch', -1, """Save n checkpoints per every epoch""")
flags.DEFINE_string('ckpt_dir', None, """Directory to save checkpoints""")
flags.DEFINE_string('profile_dir', None, """Directory to save RunMetadata""")
flags.DEFINE_string('profile_steps', None, """Comma separated porfile steps""")
flags.DEFINE_boolean('local_aggregation', True,
                     """Whether to use local aggregation or not""")
flags.DEFINE_boolean('boundary_among_servers', True,
                     """Whether to use operation placement among servers""")
flags.DEFINE_boolean('boundary_between_workers_and_servers', True,
                     """Whether to use operation placement between workers and servers""")
flags.DEFINE_string('export_graph_path', None, """export path to keep transformed graph definintion""")
flags.DEFINE_boolean('search_partitions', False, """Whether to use variable partitioning method""")
FLAGS = flags.FLAGS

def calculate_ckpt_steps():
    if FLAGS.save_n_ckpts_per_epoch > 0:
      with open(FLAGS.resource_info_file) as resource_info:
        num_workers = sum([len(w['gpus']) for w in json.load(resource_info)['worker']])
      num_words_per_iter = FLAGS.batch_size * FLAGS.num_steps * num_workers
      num_iters_per_epoch = math.ceil(language_model_graph._NUM_WORDS['train'] / num_words_per_iter / FLAGS.save_n_ckpts_per_epoch)
      save_ckpt_steps = num_iters_per_epoch if FLAGS.sync else num_iters_per_epoch * num_workers
      parallax.log.info('Save checkpoint for every %d iters' % save_ckpt_steps)
    else:
      save_ckpt_steps = FLAGS.save_ckpt_steps

    return save_ckpt_steps


def build_config():

    ckpt_config = parallax.CheckPointConfig(ckpt_dir=FLAGS.ckpt_dir,
                                            save_ckpt_steps=calculate_ckpt_steps())
    ps_config = parallax.PSConfig(replicate_variables=FLAGS.replicate_variables,
                                  protocol=FLAGS.protocol,
                                  local_aggregation=FLAGS.local_aggregation,
                                  boundary_among_servers=FLAGS.boundary_among_servers,
                                  boundary_between_workers_and_servers=\
                                  FLAGS.boundary_between_workers_and_servers)
    mpi_config = parallax.MPIConfig(mpirun_options=FLAGS.mpirun_options)
    parallax_config = parallax.Config()
    parallax_config.run_option = FLAGS.run_option
    parallax_config.average_sparse = False
    parallax_config.communication_config = parallax.CommunicationConfig(ps_config, mpi_config)
    parallax_config.ckpt_config = ckpt_config
    def get_profile_steps():
        if not FLAGS.profile_steps:
            return []
        FLAGS.profile_steps = FLAGS.profile_steps.strip()
        return [int(step) for step in FLAGS.profile_steps.split(',')]
    profile_config = parallax.ProfileConfig(profile_dir=FLAGS.profile_dir,
                                            profile_steps=get_profile_steps())
    parallax_config.profile_config = profile_config
    parallax_config.redirect_path = FLAGS.redirect_path
    parallax_config.export_graph_path = FLAGS.export_graph_path
    parallax_config.search_partitions = FLAGS.search_partitions

    return parallax_config

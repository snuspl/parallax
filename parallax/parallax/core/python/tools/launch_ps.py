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

import argparse
import sys, os
import json

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of target hosts""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of target hosts""")
tf.app.flags.DEFINE_string('job_name', '',
                           """Job name in cluster""")
tf.app.flags.DEFINE_integer('task_index', -1,
                            """Task index of the job""")
tf.app.flags.DEFINE_string('protocol', 'grpc',
                           """Server protocol: grpc, grpc+verbs, grpc+gdr""")


def main(argv=None):
    assert FLAGS.job_name == 'ps'
    tf_cluster_dict = {}

    if not FLAGS.ps_hosts == '':
        tf_cluster_dict['ps'] = []
        for ps in FLAGS.ps_hosts.split(','):
            tf_cluster_dict['ps'].append(ps)

    tf_cluster_dict['worker'] = []
    for worker in FLAGS.worker_hosts.split(','):
        tf_cluster_dict['worker'].append(worker)
    cluster = tf.train.ClusterSpec(tf_cluster_dict)

    server = tf.train.Server(cluster, job_name='ps',
                             task_index=FLAGS.task_index,
                             protocol=FLAGS.protocol)
    server.join()


if __name__ == "__main__":
    tf.app.run()

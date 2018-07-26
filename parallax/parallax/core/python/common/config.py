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

from __future__ import print_function

from parallax.core.python.common.lib import *
from parallax.core.python.common.consts import *

class PSConfig(object):
    def __init__(self,
                 protocol='grpc',
                 replicate_variables=True):
        """
        Args:
          protocol: Specifies the protocol to be used by the server.
            Acceptable values include `"grpc"`. Defaults to the value in
            `server_or_cluster_def`, if specified. Otherwise defaults to
            `"grpc"`.
          replicate_variables: Each GPU has a copy of the variables,
            and updates its copy after the parameter servers are all
            updated with the gradients from all servers. Only works with
            `sync=True`.
        """
        self.protocol = protocol
        self.replicate_variables = replicate_variables


class MPIConfig(object):
    def __init__(self,
                 use_allgatherv=False,
                 mpirun_options=''):
        """
          Args:
          use_allgatherv: Specifies whether to utilize OpenMPI `allgatherv`
            instead of NCCL 'allgather'. `use_allgatherv=False` is
            recommended by Parallax.
          mpirun_options: A string or a list of strings. Specifies the extra
            configurations for mpirun. See
            https://www.open-mpi.org/doc/v3.0/man1/mpirun.1.php#sect3 for more
            details.
        """
        self.use_allgatherv = use_allgatherv
        self.mpirun_options = self.parse_mpirun_options(mpirun_options)

    def parse_mpirun_options(self, mpirun_options):
        if isinstance(mpirun_options, str):
            return mpirun_options
        elif isinstance(mpirun_options, list):
            return ' '.join([str(option) for option in mpirun_options])
        else:
            assert False, 'mpirun_options should be a string or a list of strings'


class CommunicationConfig(object):
    def __init__(self,
                 ps_config=PSConfig(),
                 mpi_config=MPIConfig()):

        assert isinstance(ps_config, PSConfig)
        assert isinstance(mpi_config, MPIConfig)

        self.ps_config = ps_config
        self.mpi_config = mpi_config


class CheckPointConfig(object):
    def __init__(self,
                 ckpt_dir=None,
                 save_ckpt_steps=None,
                 save_ckpt_secs=None):
        """
        Args:
          ckpt_dir: The checkpoint directory to store/restore global variables.
          save_ckpt_steps: The frequency, in number of global steps, that a
            checkpoint is saved using a default checkpoint saver.
          save_ckpt_secs: The frequency, in seconds, that a checkpoint is
            saved using a default checkpoint saver.
        """
        self.ckpt_dir = ckpt_dir
        self.save_ckpt_steps = save_ckpt_steps
        self.save_ckpt_secs = save_ckpt_secs


class ParallaxConfig(object):
    def __init__(self,
                 run_option=None,
                 average_sparse=False,
                 sess_config=None,
                 redirect_path=None,
                 communication_config=CommunicationConfig(),
                 ckpt_config=CheckPointConfig()):
        """Configurable options of Parallax.

        Args:
          run_option: A string(PS or MPI). If the option is None, the faster
            communication is selected automatically.
          average_sparse: A boolean. If True, sparse parameters are updated
            by the averaged gradients over all replicas. Otherwise,
            the sum of all gradients are used.
          sess_config: tf.ConfigProto object to create the session with custom
            configurations.
          redirect_path: A string. Optional path to redirect logs as files.
          communication_config: A `CommunicationConfig` object to manage the
            configurations related to communication.
          ckpt_config: A `CheckPointConfig` object to manage the checkpoints
 
        """

        self.run_option = run_option
        self.average_sparse = average_sparse
        self.sess_config = sess_config
        self.redirect_path = redirect_path

        self.communication_config = communication_config
        self.ckpt_config = ckpt_config

        self._sync = None
        self._num_iterations = None
        self._resource_info = None

    def get_ckpt_config(self, is_test):
        if is_test:
            return CheckPointConfig()
        else:
            return self.ckpt_config

    def set_sync(self, sync):
        self._sync = sync

    @property
    def sync(self):
        return self._sync

    def set_num_iterations(self, num_iterations):
        self._num_iterations = num_iterations

    def num_iterations(self, is_test):
        if is_test:
            return NUM_ITERATIONS_FOR_TEST
        else:
            return self._num_iterations

    def set_resource_info(self, resource_info):
        self._resource_info = resource_info

    @property
    def resource_info(self):
        return self._resource_info

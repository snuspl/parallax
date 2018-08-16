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

import os

PARALLAX_RUN_OPTION = "PARALLAX_RUN_OPTION"
PARALLAX_RUN_MASTER = "PARALLAX_RUN_MASTER"
PARALLAX_RUN_MPI = "PARALLAX_RUN_MPI"
PARALLAX_RUN_PS = "PARALLAX_RUN_PS"
PARALLAX_RUN_HYBRID = "PARALLAX_RUN_HYBRID"
PARALLAX_WORKER_ID = "PARALLAX_WORKER_ID"
PARALLAX_NUM_WORKERS = "PARALLAX_NUM_WORKERS"
PARALLAX_RESOURCE_INFO = "PARALLAX_RESOURCE_INFO"
PARALLAX_MACHINE_ID = "PARALLAX_MACHINE_ID"
PARALLAX_HOSTNAME = "PARALLAX_HOSTNAME"

LOCAL_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_LAUNCH_PS_PATH = os.path.join(LOCAL_CODE_ROOT, 'tools',
                                    'launch_ps.py')

REMOTE_PARALLAX_ROOT = os.path.join('/tmp', 'parallax-%s' % os.environ['USER'])
REMOTE_LAUNCH_PS_PATH = os.path.join(REMOTE_PARALLAX_ROOT, 'launch_ps.py')
REMOTE_MPI_SCRIPT_PATH = os.path.join(REMOTE_PARALLAX_ROOT, 'mpi_run.sh')

NUM_ITERATIONS_FOR_TEST = 200
NUM_ITERATIONS_FOR_WARMUP = 200

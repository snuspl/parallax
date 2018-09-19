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

import tensorflow as tf

PARALLAX_PARTITIONS = "PARALLAX_PARTITIONS"
PARALLAX_SEARCH = "PARALLAX_SEARCH"

def get_partitions(default_num_shards, axis=0):
  if PARALLAX_PARTITIONS in os.environ:
      num_shards = int(os.environ[PARALLAX_PARTITIONS])
  else:
      num_shards = max(default_num_shards, 1)
  return num_shards
  

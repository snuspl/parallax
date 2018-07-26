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

from absl import flags
import tensorflow as tf

import benchmark_cnn

benchmark_cnn.define_flags()
flags.adopt_module_key_flags(benchmark_cnn)

FLAGS = tf.app.flags.FLAGS

def main(_):
  FLAGS.eval = True
  params = benchmark_cnn.make_params_from_flags()
  params, config = benchmark_cnn.setup(params)
  bench = benchmark_cnn.BenchmarkCNN(params)
  bench.evaluate()

if __name__ == '__main__':
  tf.app.run()

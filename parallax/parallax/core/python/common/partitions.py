# Copyright (C) 2019 Seoul National University
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
from multiprocessing.managers import BaseManager
try:
  import Queue as queue
except ImportError:
  import queue
import numpy as np
from scipy import optimize
import time

import tensorflow as tf

from parallax.core.python.common.lib import *

PARALLAX_MIN_PARTITIONS = "PARALLAX_MIN_PARTITIONS"
PARALLAX_PARTITIONS = "PARALLAX_PARTITIONS"
PARALLAX_SEARCH = "PARALLAX_SEARCH"

#TODO: support partitioning for multiple partitioners
#      with different number of partitions
def get_partitioner(min_num_partitions):
    """Return tf.fixed_size_partitioner with num_partitions
       that determined by Parallax.
   
    Args:
      min_num_partitions: A minimum (default) number of partitions 
                          without memory exception.
    """

    if PARALLAX_MIN_PARTITIONS not in os.environ:
       os.environ[PARALLAX_MIN_PARTITIONS] = str(min_num_partitions)

    if PARALLAX_PARTITIONS in os.environ:
        partitions = int(os.environ[PARALLAX_PARTITIONS])
    else:
        partitions = min_num_partitions
    return tf.fixed_size_partitioner(partitions)

class PartitionStatCollector(object):

    def __init__(self, p_to_test, address):
        self.p_to_test = p_to_test
        self.address = address
        self.prev_p = None
        self.prev_exec_time = None
        self.exec_time_list = []
        self.p_list = []
        self.start = None
        self.min_partitions = int(os.environ[PARALLAX_MIN_PARTITIONS])

    def setup_manager(self):
        if self.start is None:
            self.start = time.time()
        self.m = BaseManager(address=self.address, authkey='parallax_auth')
        queue = queue.Queue()
        BaseManager.register('queue', callable=lambda:queue)
        self.m.start()
        return self.m

    def recv_exec_time(self, processes, cleanup, num_required):
        stop = False
        worker_exec_times = []
        all_alive = True
        while len(worker_exec_times) != num_required and all_alive:
            time.sleep(10)
            q = self.m.queue()
            while q.qsize() > 0:
                worker_exec_times.append(q.get())
 
            for p in processes:
                if p.poll() is not None:
                    all_alive = False
                    break

        cleanup(None, None)
        time.sleep(10)

        if all_alive:
            curr_p = self.p_to_test
            curr_exec_time = np.mean(worker_exec_times)
            self.p_list.append(curr_p)
            self.exec_time_list.append(curr_exec_time)

            if self.prev_p:
                if self.prev_exec_time < curr_exec_time:
                    # decrease or stop
                    if self.prev_p > curr_p:
                        stop = True
                    else:
                        # search the oposite partitions
                        self.p_to_test = min(self.p_list) / 2
                else:
                    assert (self.prev_exec_time / curr_exec_time) > 1
                    # keep increase or keep decrease
                    if self.prev_p < curr_p:
                        self.p_to_test *= 2
                    else:
                        self.p_to_test /= 2

                if self.p_to_test < self.min_partitions:
                    stop = True
            else:
                # increase first
                self.p_to_test *= 2

            self.prev_p = curr_p
            self.prev_exec_time = curr_exec_time
        else:
            # communication error when num partitions is small
            if self.prev_p:
                stop = True
            else:
                self.p_to_test *= 2
                self.min_partitions = self.p_to_test

        if stop:
            end = time.time()
            self.p_to_test = self._find_optimal_p()
            parallax_log.info('optimal partitions: %d, search time: %d secs' % \
                (self.p_to_test, end - self.start))
            print('optimal partitions: %d, search time: %d secs' % \
                (self.p_to_test, end - self.start))

        return not stop, self.p_to_test

    def _find_optimal_p(self):
        parallax_log.info('start finding optimal p')
        print('start finding optimal p')
        parallax_log.info(self.p_list)
        print(self.p_list)
        parallax_log.info(self.exec_time_list)
        print(self.exec_time_list)
        
        if len(self.p_list) < 3:
            min_exec_time = min(self.exec_time_list)
            return self.p_list[self.exec_time_list.index(min_exec_time)]
            
        max_time = float(max(self.exec_time_list))
        exec_times = [t / max_time for t in self.exec_time_list]

        fitfunc = lambda n, a, b, c: b / n + a * (n - 1) + c
        p, pcov = optimize.curve_fit(fitfunc, np.array(self.p_list), np.array(exec_times))

        min_p = min(self.p_list)
        max_p = max(self.p_list)

        min_exec_time = None
        optimal_p = None
        for i in range(min_p, max_p + 1):
            prediction = fitfunc(i, p[0], p[1], p[2])

            if min_exec_time is None or min_exec_time > prediction:
                min_exec_time = prediction
                optimal_p = i

        return optimal_p

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

import contextlib
from multiprocessing.managers import BaseManager
import os
import Queue
import threading
import time

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as print_mdl
from tensorflow.python.client import session
from tensorflow.python.util import compat

from parallax.core.python.common.partition import *

COLLECT_STAT_START = 50
COLLECT_STAT_END = 100
   
def _parallax_init(self, target='', graph=None, config=None):
    """Overwrites the session.__init__."""
    self._init_internal(target, graph, config)  # pylint: disable=protected-access

def _parallax_run(self,
                  fetches,
                  feed_dict=None,
                  options=None,
                  run_metadata=None):
    
    fetches = self.parallax_session_context._convert_fetch(fetches)
    feed_dict = self.parallax_session_context._convert_feed(feed_dict)

    if (not self.parallax_session_context._send_exec_time and 
        (self.parallax_session_context._profile_dir is None 
        or self.parallax_session_context._profile_steps is None)):
        return self._run_internal(fetches, feed_dict)

    with self.parallax_session_context._new_step() as state:
        step, locked = state
        if locked and self.parallax_session_context._send_exec_time:
            start_step = self.parallax_session_context._start_step
            relative_step = step - start_step
            if COLLECT_STAT_START <= relative_step and relative_step <= COLLECT_STAT_END:
                start = time.time()
                ret = self._run_internal(fetches, feed_dict)
                end = time.time()
                self.parallax_session_context._exec_time += (end - start)
                if step == COLLECT_STAT_END:
                    host = self.parallax_session_context._master['hostname']
                    port = int(self.parallax_session_context._master['port'][0])
                    BaseManager.register('queue')
                    m = BaseManager(address=(host, port), authkey='parallax_auth')
                    m.connect()
                    queue = m.queue()
                    queue.put(self.parallax_session_context._exec_time)
            else:
                ret = self._run_internal(fetches, feed_dict)
        elif locked and self.parallax_session_context._is_profile_step(step):
            if not run_metadata:
                run_metadata = tf.RunMetadata()
            if not options:
                options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                old_trace_level = options.trace_level
            else:
                old_trace_level = options.trace_level
                options.trace_level = tf.RunOptions.FULL_TRACE
                
            ret = self._run_internal(
                fetches, feed_dict, options, run_metadata)
            self.parallax_session_context._dump_profile(
                run_metadata, 'run_meta_%d' % step)
            options.trace_level = old_trace_level
        else:
            ret = self._run_internal(fetches, feed_dict)

    return ret          
                      
class ParallaxSessionContext(object):
    """A context that wraps session for Parallax.
       
    This class references tf.contrib.tfprof.ProfileContext class.
    """
   
    def __init__(self,
                 step,
                 profile_dir,
                 profile_steps,
                 replica_dict,
                 num_replicas_per_worker,
                 master):
        """Constructs an `ParallaxSessionContext` instance.

        Args:
          profile_dir: Directory to store profiles.
          profile_steps: A list of steps for tracing and saving as a file.
          replica_dict : A dictionary to map old tensor(operation) name
            to new tensor(operation) names.
          num_replicas_per_worker : Number of replicas per worker.
        """
        self._start_step = step
        self._step = step
        self._profile_dir = profile_dir
        self._profile_steps = profile_steps
        self._replica_dict = replica_dict
        self._num_replicas_per_worker = num_replicas_per_worker
        self._send_exec_time = os.environ[PARALLAX_SEARCH] == 'True'
        self._exec_time = 0
        self._master = master

        for key, values in self._replica_dict.items():
            if len(values) == 1:
                item = values[0]
                self._replica_dict[key] = [item for _ in
                    range(self._num_replicas_per_worker)]
        self._lock = threading.Lock()
    
    @contextlib.contextmanager
    def _new_step(self):
        acquired = self._lock.acquire(False)
        yield (self._step, acquired)
        self._step += 1
        if acquired:
            self._lock.release()
 
    def _is_profile_step(self, step):
      if step in self._profile_steps:
        return True
      return False

    def _dump_profile(self, metadata, basename):
      if not tf.gfile.Exists(self._profile_dir):
          tf.gfile.MakeDirs(self._profile_dir)
      with tf.gfile.Open(os.path.join(self._profile_dir, basename), 'wb') as f:
          f.write(metadata.SerializeToString())

    def _read_converted_names(self, target):
        if isinstance(target, compat.bytes_or_text_types):
            target_name = target
        else:
            target_name = target.name
        if target_name in self._replica_dict:
            return self._replica_dict[target_name]
        else:
            return target
     
    def _convert_fetch(self, fetch):
        if fetch is None:
            raise TypeError('Fetch argument %r has invalid type %r' % (fetch,
                                                                 type(fetch)))
        elif isinstance(fetch, (list, tuple)):
            return [self._convert_fetch(f) for f in fetch]
        elif isinstance(fetch, dict):
            keys = list(fetch.keys())
            values = [self._convert_fetch(f) for f in fetch.values()]
            return dict(zip(keys, values))
        else:
            if isinstance(fetch, tf.SparseTensor):
                return [tf.SparseTensor(self._replica_dict[fetch.indices][i],
                                        self._replica_dict[fetch.values][i],
                                        self._replica_dict[fetch.dense_shape][i]) 
                           for i in range(self._num_replicas_per_worker)]
            elif isinstance(fetch, tf.IndexedSlices):
                return [tf.IndexedSlices(
                           self._replica_dict[fetch.values][i],
                           self._replica_dict[fetch.indices][i],
                           None if fetch.dense_shape is None \
                                else self._replica_dict[fetch.dense_shape][i]) 
                               for i in range(self._num_replicas_per_worker)]
            else:
                return self._read_converted_names(fetch)

    def _convert_feed(self, feed_dict):

        def _feed_fn(feed):
            for tensor_type, _, _, feed_fn in session._REGISTERED_EXPANSIONS:
                if isinstance(feed, tensor_type):
                    return feed_fn(feed)
            raise TypeError('Feed argument %r has invalid type %r' % (feed,
                                                                   type(feed)))
        if feed_dict:
            new_feed_dict = {}
            for feed, feed_val in feed_dict.items():
                if isinstance(feed, compat.bytes_or_text_types):
                    new_feeds = self._read_converted_names(feed)
                    if isinstance(new_feeds, list):
                        for i in range(self._num_replicas_per_worker):
                            new_feed_dict[new_feeds[i]] = feed_val[i]
                    else:
                        new_feed_dict[new_feeds] = feed_val
                else:
                    for subfeed in _feed_fn(feed):
                        new_subfeeds = self._read_converted_names(subfeed)
                        if isinstance(new_subfeeds, list):
                            for i in range(self._num_replicas_per_worker):
                                new_feed_dict[new_subfeeds[i]] = feed_val[i]
                        else:
                            new_feed_dict[new_subfeeds] = feed_val
            return new_feed_dict
        else:
            return feed_dict
   
    def set_parallax_session_context(self):
      self.old_run = getattr(session.BaseSession, 'run', None)
      self.old_init = getattr(session.BaseSession, '__init__', None)
      if not self.old_run:
        raise tf.errors.InternalError(None, None, 'BaseSession misses run method.')
      elif not self.old_init:
        raise tf.errors.InternalError(None, None,
                                   'BaseSession misses __init__ method.')
      elif getattr(session.BaseSession, '_run_internal', None):
        raise tf.errors.InternalError(None, None,
                                   'Already in context or context not cleaned.')
      elif getattr(session.BaseSession, '_init_internal', None):
        raise tf.errors.InternalError(None, None,
                                   'Already in context or context not cleaned.')
      else:
        setattr(session.BaseSession, 'run', _parallax_run)
        setattr(session.BaseSession, '__init__', _parallax_init)
        setattr(session.BaseSession, '_run_internal', self.old_run)
        setattr(session.BaseSession, '_init_internal', self.old_init)
        setattr(session.BaseSession, 'parallax_session_context', self)
        return self

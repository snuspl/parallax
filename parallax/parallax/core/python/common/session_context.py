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
import os
import threading

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as print_mdl
from tensorflow.python.client import session
from tensorflow.python.util import compat

def _parallax_elem_fetch_init(self, fetches, contraction_fn):
    """Overwrites the _ElementFetchMapper.__init__."""
    
    self._unique_fetches = []
    for fetch in fetches:
        try:
            self._unique_fetches.append(
                self.parallax_session_context._convert_fetch(fetch))
        except TypeError as e:
            raise TypeError('Fetch argument %r has invalid type %r, '
                            'must be a string or Tensor. (%s)' %
                            (fetch, type(fetch), str(e)))
        except ValueError as e:
            raise ValueError('Fetch argument %r cannot be interpreted as a '
                             'Tensor. (%s)' % (fetch, str(e)))
        except KeyError as e:
            raise ValueError('Fetch argument %r cannot be interpreted as a '
                             'Tensor. (%s)' % (fetch, str(e)))
    self._contraction_fn = contraction_fn 
   
def _parallax_init(self, target='', graph=None, config=None):
    """Overwrites the session.__init__."""
    self._init_internal(target, graph, config)  # pylint: disable=protected-access

def _parallax_run(self,
                  fetches,
                  feed_dict=None,
                  options=None,
                  run_metadata=None):

    if (self.parallax_session_context._profile_dir is None 
        or self.parallax_session_context._profile_steps is None):
        return self._run_internal(fetches, feed_dict)

    with self.parallax_session_context._new_step() as state:
        step, locked = state
        if locked and self.parallax_session_context._is_profile_step(step):
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

    Args:
        profile_dir: Directory to store profiles.
        profile_steps: A list of steps for tracing and saving as a file.
        replica_dict : A dictionary to map old tensor(operation) name 
            to new tensor(operation) names.
    """
   
    def __init__(self,
                 step,
                 profile_dir,
                 profile_steps,
                 replica_dict):

        self._step = step
        self._profile_dir = profile_dir
        self._profile_steps = profile_steps
        self._replica_dict = replica_dict

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
      with tf.gfile.Open(os.path.join(self._profile_dir, basename), 'w') as f:
          f.write('%s' % metadata)

    def _convert_fetch(self, fetch):
      if isinstance(fetch, compat.bytes_or_text_types):
          fetch_name = fetch
      else:
          fetch_name = fetch.name

      if fetch_name not in self._replica_dict:
        raise ValueError('Fetch argument %s does not exist in the graph' 
            % fetch)      
      else:
          # TODO(soojeong) : Only first replica item is fetchable for now
          #                  even though num_replicas_per_workers > 1.
          #                  It can be solved after changing  
          #                  num_replicas_per_workers as 1 for PS.
          return tf.get_default_graph().as_graph_element(
              self._replica_dict[fetch_name][0])

    def __call__(self):
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
        setattr(session._ElementFetchMapper, '__init__', _parallax_elem_fetch_init)
        setattr(session._ElementFetchMapper, 'parallax_session_context', self)
        setattr(session.BaseSession, 'run', _parallax_run)
        setattr(session.BaseSession, '__init__', _parallax_init)
        setattr(session.BaseSession, '_run_internal', self.old_run)
        setattr(session.BaseSession, '_init_internal', self.old_init)
        setattr(session.BaseSession, 'parallax_session_context', self)
        return self

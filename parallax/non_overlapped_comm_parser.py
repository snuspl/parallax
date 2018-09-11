import os
import re
import json
import tensorflow as tf
import time
import numpy as np

def get_computation_times(run_metadata):
  comp_times = []
  
  for dev_stat in run_metadata.step_stats.dev_stats:
    if 'stream' not in dev_stat.device:
      for node_stat in dev_stat.node_stats:
        if node_stat.node_name == 'RecvTensor' or \
          node_stat.node_name.startswith('HorovodAll'):
          continue
        start = node_stat.all_start_micros
        end = node_stat.all_end_rel_micros + start
        to_merge_index = []

        search_start = -1
        for i in range(len(comp_times)):
          s, e = comp_times[i]
          if end < s or (s <= start and e >= start):
            break
          search_start = i

        if search_start < 0:
          comp_times.insert(0, (start, end))
          continue

        for i in range(search_start, len(comp_times)):
          s, e = comp_times[i]
          if end < s:
            break
          elif start <= e:
            to_merge_index.append(i)
            start = min(s, start)
            end = max(e, end)
          else:
            search_start += 1
          
        assert len(to_merge_index) <= len(comp_times)
        if to_merge_index:
          for i in range(len(to_merge_index)):
            del comp_times[to_merge_index[0]]
        comp_times.insert(search_start, (start, end))  
  return comp_times
          
def parse_timeline_label(label):
  match = re.match(r'\[(.*)\] edge_([0-9]+)_(.*) from (.*) to (.*)', label)
  data, edge_num, tensor, from_d, to_d = match.groups()
  def _as_bytes(d):
    if 'MB' in d:
      return float(d.split('MB')[0]) * 1048576.0
    else:
      return float(d.split('B')[0])

  return _as_bytes(data), tensor 

def parse_runmetadata(run_metadata):
  comm_info = []

  for dev_stat in run_metadata.step_stats.dev_stats:
    if 'stream' not in dev_stat.device:
      for node_stat in dev_stat.node_stats:
        if node_stat.node_name == 'RecvTensor':
          comm_type = 'PS'
          bytes, tensor = parse_timeline_label(node_stat.timeline_label)
        elif node_stat.node_name.startswith('HorovodAll'):
          allreduce = node_stat.node_name.startswith('HorovodAllreduce')
          comm_type = 'MPI(%s)' % 'allreduce' if allreduce else 'allgather'
          bytes = -1
          tensor = node_stat.node_name
        else:
          continue
        start = node_stat.all_start_micros
        duration = node_stat.all_end_rel_micros
        comm_info.append((tensor, bytes, start, duration, comm_type))  
  return comm_info
      
def parse_comm(data_dir, parsed_data_dir):
  if not os.path.exists(parsed_data_dir):
    os.makedirs(parsed_data_dir)

  apps = os.listdir(data_dir)
  for app in apps:
    if app == 'parsed_thp' or app == 'parsed_comm':
      continue

    parsed_app_dir = os.path.join(parsed_data_dir, app)
    if not os.path.exists(parsed_app_dir):
      os.makedirs(parsed_app_dir)

    app_dir = os.path.join(data_dir, app)
    machines = os.listdir(app_dir) 
    machines.sort()
    for machine_ in machines:
      machine_dir = os.path.join(app_dir, machine_)
      parsed_machine_dir = os.path.join(parsed_app_dir, machine_)
      if not os.path.exists(parsed_machine_dir):
        os.makedirs(parsed_machine_dir)
#      write_file = open(os.path.join(parsed_machine_dir, 'parsed_comm'), 'w+')

      num_replicas = int(machine_.split('M')[1]) * 6
#      write_file.write('Tensor, DataSize(bytes), CommStart(usecs), CommTime(usecs), CommType\n')
      partitions = os.listdir(machine_dir)
      partitions.sort(key=lambda x: int(x.split('P')[1]) if 'P' in x else x)
      for partition_ in partitions:
        partition_dir = os.path.join(machine_dir, partition_)
        if not partition_.startswith('P') or not os.path.isdir(partition_dir):
          continue
#        write_file.write('%s\n' % partition_)

        if not os.path.exists(os.path.join(partition_dir, 'profile', 'run_meta_410')):
          continue

        print(os.path.join(partition_dir, 'profile', 'run_meta_410'))
        run_metadata = tf.RunMetadata()
        with open(os.path.join(partition_dir, 'profile', 'run_meta_410'), 'rb') as f:
          run_metadata.MergeFromString(f.read())

        comp_times = get_computation_times(run_metadata)
        print([(x - comp_times[0][0], y - comp_times[0][0]) for x,y in comp_times])
        comm_info = parse_runmetadata(run_metadata)
        print(comm_info)        
        sys.exit(-1)
#        comm_logs = remove_overlap(comp_series, comm_info)
#        for comm_log in comm_logs:
#          write_file.write('%s\n' % comm_log)
#      write_file.close()
          

if __name__ == '__main__':
  apps = ['lm1b', 'nmt']
  data_dir = '/home/soojeong/partitions_exp'
  parsed_data_dir = '/home/soojeong/partitions_exp/parsed_comm'

  parse_comm(data_dir, parsed_data_dir)

import os
import re
import json
import tensorflow as tf
import time
import numpy as np

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
  comm_logs = []  

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
        comm_logs.append('%s, %d, %d, %d, %s' % (tensor, bytes, start, duration, comm_type))  
  return comm_logs
      
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
      write_file = open(os.path.join(parsed_machine_dir, 'parsed_comm'), 'w+')

      num_replicas = int(machine_.split('M')[1]) * 6
      write_file.write('Tensor, DataSize(bytes), CommStart(usecs), CommTime(usecs), CommType\n')
      partitions = os.listdir(machine_dir)
      partitions.sort(key=lambda x: int(x.split('P')[1]) if 'P' in x else x)
      for partition_ in partitions:
        partition_dir = os.path.join(machine_dir, partition_)
        if not partition_.startswith('P') or not os.path.isdir(partition_dir):
          continue
        write_file.write('%s\n' % partition_)

        if not os.path.exists(os.path.join(partition_dir, 'profile', 'run_meta_410')):
          continue

        run_metadata = tf.RunMetadata()
        with open(os.path.join(partition_dir, 'profile', 'run_meta_410'), 'rb') as f:
          run_metadata.MergeFromString(f.read())

        comm_logs = parse_runmetadata(run_metadata)
        for comm_log in comm_logs:
          write_file.write('%s\n' % comm_log)
      write_file.close()
          

if __name__ == '__main__':
  apps = ['lm1b', 'nmt']
  data_dir = '/home/soojeong/partitions_exp'
  parsed_data_dir = '/home/soojeong/partitions_exp/parsed_comm'

  parse_comm(data_dir, parsed_data_dir)

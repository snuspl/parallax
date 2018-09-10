import os
import re
import json
import time
import numpy as np

def parse_thp(data_dir, parsed_data_dir):
  if not os.path.exists(parsed_data_dir):
    os.makedirs(parsed_data_dir)

  apps = os.listdir(data_dir)
  for app in apps:
    if app == 'parsed_thp' or app == 'parsed_comm':
      continue

    if not os.path.exists(os.path.join(parsed_data_dir, app)):
      os.makedirs(os.path.join(parsed_data_dir, app))
    
    write_file = open(os.path.join(parsed_data_dir, app, 'parsed_thp'), 'w+')

    app_dir = os.path.join(data_dir, app)
    machines = os.listdir(app_dir)
    machines.sort()
    for machine_ in machines:
      machine_dir = os.path.join(app_dir, machine_)
      num_replicas = int(machine_.split('M')[1]) * 6
      write_file.write('%s\n' % machine_) 
      partitions = os.listdir(machine_dir)
      partitions.sort(key=lambda x: int(x.split('P')[1]) if 'P' in x else x)
      for partition_ in partitions:
        partition_dir = os.path.join(machine_dir, partition_)
        if not partition_.startswith('P') or not os.path.isdir(partition_dir):
          continue
        if not os.path.exists(os.path.join(partition_dir, 'train_log')):
          continue
        with open(os.path.join(partition_dir, 'train_log'), 'r') as f:
          wps_list = []
          for line in f: 
            if app == 'lm1b':
              # Iteration 201, time = 67.32s, wps = 3803, train loss = 6.3289
              match = re.match('(.*)Iteration (.*), (.*), wps = (.*), (.*)', line)
              if match:
                _, step, _, wps, _ = match.groups()
                if int(step) >= 300 and int(step) <= 410:
                  wps_list.append(int(wps))
            elif app == 'nmt':
              #step 100 lr 1 step-time 4.19s wps 1.71K ppl 14313.75 gN 65.55 bleu 0.00, Mon Sep 10 14:16:17 2018 
              match = re.match('(.*)step ([0-9]+) lr(.*) wps (.*) ppl(.*)', line)
              if match:
                _, step, _, wps, _ = match.groups()
                if int(step) >= 300 and int(step) <= 410:
                  wps_num = float(wps.split('K')[0]) * 1000
                  wps_list.append(wps_num)
          if wps_list:
            total_wps = np.mean(wps_list) * num_replicas
            write_file.write('%s: wps = %d\n' % (partition_, total_wps))
    write_file.close() 

if __name__ == '__main__':
  apps = ['lm1b', 'nmt']
  data_dir = '/home/soojeong/partitions_exp'
  parsed_data_dir = '/home/soojeong/partitions_exp/parsed_thp'

  parse_thp(data_dir, parsed_data_dir)

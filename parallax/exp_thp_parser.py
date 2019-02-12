import os
import re
import json
import time
import numpy as np

def parse_thp(data_dir, parsed_data_dir):
  if not os.path.exists(parsed_data_dir):
    os.makedirs(parsed_data_dir)

  print(data_dir)
  apps = os.listdir(data_dir)
  print(apps)
  for app in apps:
    if app == 'parsed_thp' or app == 'parsed_comm':
      continue

    print(app)
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
      optimal_partition = None
      search_time = 0
      optimal_partition_exec_time = None
      for trial in range(1):
        trial_dir = os.path.join(machine_dir, 'Test-%d' % trial)
        print(trial_dir)
         
        if not os.path.exists(os.path.join(trial_dir, 'train_log')):
          print('train log is missing in %s' % trial_dir)
          continue
        with open(os.path.join(trial_dir, 'train_log'), 'r') as f:
          wps_list = []
          times = []
          line=f.readline()
          while line:
            optimal_match = re.match('start finding optimal p', line)
            find_opt = False
            if optimal_match:
              partitions = f.readline()
              write_file.write('sampled_partitions: %s\n' % partitions)
              exec_times = f.readline()
              write_file.write('sampled_step_time(secs): %s\n' % exec_times)
              write_file.write(f.readline())
            elif app == 'lm1b':
              # Iteration 201, time = 67.32s, wps = 3803, train loss = 6.3289
              match = re.match('(.*)Iteration (.*), time = (.*)s, wps = (.*), (.*)', line)
              if match:
                _, step, time, wps, _ = match.groups()
                if int(step) >= 300 and int(step) <= 410:
                  wps_list.append(int(wps))
                  times.append(float(time))
            elif app == 'nmt':
              #step 100 lr 1 step-time 4.19s wps 1.71K ppl 14313.75 gN 65.55 bleu 0.00, Mon Sep 10 14:16:17 2018 
              match = re.match('(.*)step ([0-9]+) lr(.*) step-time (.*)s wps (.*) ppl(.*)', line)
              if match:
                _, step, _, time, wps, _ = match.groups()
                if int(step) >= 250 and int(step) <= 410:
                  wps_num = float(wps.split('K')[0]) * 1000
                  wps_list.append(wps_num)
                  times.append(float(time))
            line=f.readline()  
          if wps_list:
            total_wps = np.mean(wps_list) * num_replicas
            write_file.write('%s: wps = %d, time = %f\n' % (optimal_partition, total_wps, np.mean(times)))
    write_file.close() 

if __name__ == '__main__':
  apps = ['nmt']
  data_dir = '/home/soojeong/nmt_label_smoothing_fast_shard_thp'
  parsed_data_dir ='/home/soojeong/nmt_label_smoothing_fast_shard_thp/parsed_thp'

  parse_thp(data_dir, parsed_data_dir)

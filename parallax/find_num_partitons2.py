import os
import re
from scipy import optimize
import numpy as np
import collections

def merge_overlap(data, to_add):
  to_add_start = to_add[1]
  to_add_end = to_add[2]
  bytes = to_add[0]

  insert_index = -1
  def _overlap(start1, end1, start2, end2):
    return not (end1 < start2) or (end2 < start1) 

  to_merge = []  
  for i in range(len(data)):
    s = data[i][1]
    e = data[i][2]
    if to_add_end < s:
      break
    elif _overlap(to_add_start, to_add_end, s, e):
      to_merge.append(i)
      to_add_start = min(to_add_start, s)
      to_add_end = max(to_add_end, e)
      bytes += data[i][0]

  if to_merge:
    insert_index = min(to_merge)
    for i in range(len(to_merge)):
      del data[insert_index]

  if insert_index < 0:
    data.insert(0, to_add)
  else:
    data.insert(insert_index, to_add)
    
def parse_comm(comm_file_path):
  comp_times = {}
  data = {}
  with open(comm_file_path, 'r') as f:
    num_partitions = None
    for line in f:
      if line.startswith('P'):
        num_partitions = int(line.split('P')[1])
      elif line.startswith('('):
        assert num_partitions
        comp_times[num_partitions] = []
        data[num_partitions] = []
        for comp_time in line.split('), '):
          split = comp_time.strip('(').strip(')\n').split(', ')
          comp_times[num_partitions].append((int(split[0]), int(split[1])))
      elif num_partitions:
        comm_log = line.strip().split(', ')
        if True:
          tensor_name = comm_log[0]
          bytes = int(comm_log[1])
          start_time = int(comm_log[2])
          duration = int(comm_log[3])
          merge_overlap(data[num_partitions], (bytes, start_time, start_time+duration))
 
  return comp_times, data   

def find_partitons(data):
  partitions = []
  data_size = []
  comm_time = []
  for p, d_list in data.items():
    partitions.append(p)
    d_size = 0
    c_time = 0
    for d in d_list:
      d_size += d[0]
      c_time += (d[2] - d[1])
    data_size.append(d_size)
    comm_time.append(c_time)
  print(partitions)
  print(data_size)
  print(comm_time)

  fitfunc = lambda p, n, d: p[0] * (n -1) + p[1] * 1 / n 
  errfunc = lambda p, n, d, y :(fitfunc(p, n, d) - y)
  p0 = np.random.rand(2)
  p, success = optimize.leastsq(errfunc, p0, args=(np.array(partitions), np.array(data_size), np.array(comm_time)))
  print(p)
  
  min_partitions = np.min(partitions)
  max_partitions = np.max(partitions)

  min_comm_time = -1
  optimal_partitions = -1
  for i in range(min_partitions, max_partitions+1):
    total_comm_time = 0
    for d in data_size:
      estimated_time = fitfunc(p, i, d)
      #print('data size: %d, partitions: %d, estimated_comm_time: %d' % (d, i, estimated_time))
      total_comm_time += estimated_time

    if min_comm_time < 0 or total_comm_time < min_comm_time:
      min_comm_time = total_comm_time
      optimal_partitions = i

  print('optimal partitions: %d, estimated total comm time : %d' % (optimal_partitions, min_comm_time))
   
def find_partitions2(partitions, throughputs):
  min_t = min(throughputs)
  throughputs = [float(min_t)/float(t) for t in throughputs]

  real_data = {}
  for i in range(len(partitions)):
    real_data[partitions[i]] = throughputs[i]
     
  fitfunc = lambda n, a, b, c: b / n + a * (n - 1) + c
  #errfunc = lambda p, n, y :(fitfunc(p, n) - y)
  #p0 = np.random.rand(2)
  p, pcov = optimize.curve_fit(fitfunc, np.array(partitions), np.array(throughputs))
  #p, success = optimize.leastsq(errfunc, p0, args=(np.array(partitions), np.array(throughputs)))
  #print(p)
  #print(pcov)
  
  min_partitions = np.min(partitions)
  max_partitions = np.max(partitions)

  min_exec_time = -1
  optimal_partitions = -1
  errors = []
  for i in range(min_partitions, max_partitions+1):
    estimated_time = fitfunc(i, p[0], p[1], p[2])
    if i in real_data:
      #print('%f, %f' % (real_data[i], estimated_time))
      errors.append(abs(real_data[i] - estimated_time) / real_data[i])
    #print('%d, %d' % (i, estimated_throughput)) 
    if min_exec_time < 0 or min_exec_time > estimated_time:
      min_exec_time = estimated_time
      optimal_partitions = i

  print('optimal partitions: %d, estimated time : %f' % (optimal_partitions, min_exec_time))
  return np.mean(errors)

if __name__ == '__main__':
  thp_dir = '/home/soojeong/partitions_exp/parsed_thp'
  def _find(m, partitions, thp_list):
      print('find optimal partitons for machine %d' % m)
      new_p = []
      new_t = []
      prev = None
      twice = partitions[0]
      for i in range(len(partitions)):
          while partitions[i] > twice:
                twice *= 2

          if partitions[i] != twice:
                continue

          twice *= 2
          new_p.append(partitions[i])
          new_t.append(thp_list[i])
              
          if prev is not None and prev >= thp_list[i]:
              break
          prev = thp_list[i]

      print(new_p)
      print(new_t)
      error = find_partitions2(new_p, new_t)
      print('sampling prediction error : %f' % error)

  for app in os.listdir(thp_dir):
    print(app)
    app_parsed_thp = os.path.join(thp_dir, app, 'parsed_thp')
    with open(app_parsed_thp) as f:
      partitions = []
      thp_list = []
      for l in f:
        if l.startswith('M'):
          if partitions:
            _find(m, partitions,thp_list)
          m = int(l.split('M')[1])
          del partitions[:]
          del thp_list[:]
        else:
          p = int(l.split(' ')[0].split('P')[1].split(':')[0])
          t = float(l.split(' ')[3])
          partitions.append(p)
          thp_list.append(t)

      if partitions:
        _find(m, partitions, thp_list)   
  # data is a list of (name, #partitons, data_size, comm_time, start, end)
  #comp_times, data = parse_comm(comm_file_path)
  #data = collections.OrderedDict(sorted(data.items()))
  #new_data = remove_overlap(comp_times, data)
  #print(data)
  #find_partitons(data)
  # LM M8
  #r = find_partitons2(partitions=[2,4,8,16,32,64,128], throughputs=[62291, 109056, 190609, 279107, 299567, 303882, 286200])
  #print(r)
  # NMT M4
  #r = find_partitons2(partitions=[2,4,8,16,32], throughputs=[105720, 119880, 128160, 133080, 130680])
  #print(r)


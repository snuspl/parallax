import os
import json
import time

def create_resource_info(machine_dir, num_machine, elsa_set):
  resource_file_path = os.path.join(machine_dir, 'resource_info')
  with open(resource_file_path, 'w') as resfile:
    for i in range(num_machine):
      resfile.write('%s:0,1,2,3,4,5\n' % elsa_set[i])
  return resource_file_path

def run_parallax_test(data_dir, apps, machine_nums, partitions, elsa_set, vocab_sizes):
  # create data_dir
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  for app in apps:
    app_dir = os.path.join(data_dir, app)
    if not os.path.exists(app_dir):
      os.makedirs(app_dir)

    for num_machine in machine_nums:
      machine_dir = os.path.join(app_dir, 'M%d' % num_machine)
      if not os.path.exists(machine_dir):
        os.makedirs(machine_dir)
      resource_file_path = create_resource_info(machine_dir, num_machine, elsa_set)

      for vocab_size in vocab_sizes:
        partition_dir = os.path.join(machine_dir, 'vocab_%d' %vocab_size)
        if not os.path.exists(partition_dir):
          os.makedirs(partition_dir)
        #else:
        #  continue
 
        driver_args = json.load(open('driver_args.json'))
        testset = driver_args[app]
        cmd = 'python %s ' % testset['parallax_driver']
        cmd += ' '.join(testset['args'])
        cmd += ' --mpirun_options=\'-mca btl ^openib -mca btl_tcp_if_include enp129s0f0 -mca pml ob1\''
        cmd += ' --run_option=HYBRID'
        #cmd += ' --profile_steps=410,411,412,413,414,415,416,417,418,419,420 --profile_dir=%s' % os.path.join(partition_dir, 'profile')
        cmd += ' --max_steps=410'
        cmd += ' --resource_info_file=%s' % resource_file_path
        cmd += ' --protocol=grpc+verbs'
        if app == 'lm1b':
          #cmd += ' --datadir=/home/soojeong/parallax/parallax/parallax/examples/lm1b/lm1b_synthetic_top%d_words' % vocab_size
          #cmd += ' --deterministic=True' 
          cmd += ' --search_partitions=False --num_variable_shards=33 --use_synthetic=False'
          cmd += ' --num_steps=%s' % vocab_size
          #cmd += ' --search_partitions=True'
          cmd += ' --vocab_size_limit=%d' % 15360
        elif app == 'nmt':
	  cmd += ' --train_prefix=/home/soojeong/parallax/parallax/parallax/examples/nmt/nmt_synthetic_top15360_len%d_words' % vocab_size
          cmd += ' --search_partitions=False --num_embeddings_partitions=43 --vocab_size_limit=122880'
        #if vocab_size != 500:
        #    cmd += ' --search_partitions=True'
        #else:
        cmd += ' >> %s/train_log' % partition_dir
        print('=' * 60)
        print('Start the following COMMAND')
        print(cmd)
        print('=' * 60)  
        os.system('echo %s >> %s/command' % (cmd, partition_dir))
        os.system(cmd)
        time.sleep(20)   

if __name__ == '__main__':
  apps = ['lm1b']
  data_dir = '/home/soojeong/alpha_exp_vocab_15360_state_128'
  machine_nums = [8]
  vocab_sizes = [1]
  vocab_sizes.reverse()
  test_count = 1

  # Get elsa indices to use from user.
  # e1 will be used for 1 machine exp.
  # e1,e2 will be used for 2 machine exp.
  # e1,e2,e3,e4 will be used for 4 machine exp.
  # e1,e2,e3,e4,e5,e6,e7,e8 will be used for 8 machine exp.
  #elsa_set = raw_input('Input 8 elsa indices(format: 1,2,3,4,5,6,7,8): ')
  elsa_set = '3,1,2,5,6,7,8,10'
  elsa_set = ['elsa-' + '%02d'%int(elsa_idx) + '-ib0' for elsa_idx in elsa_set.split(',')]
  print(elsa_set)

  run_parallax_test(data_dir, apps, machine_nums, test_count, elsa_set, vocab_sizes)

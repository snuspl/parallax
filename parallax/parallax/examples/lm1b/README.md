# LM-1B 
LM-1B implements the LSTM language model described in [LM](https://arxiv.org/abs/1602.02410). 
The original code comes from https://github.com/rafaljozefowicz/lm, which supports 
synchronous training with multiple GPUs. We change the code as single GPU code, and 
then apply parallax auto-parallelization for multi-GPU, multi-machine with synchronous 
or asynchronous training.

## Dataset
* [1B Word Benchmark Dataset](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark)

## To Run
Set your resource information in the `resource_info` file.

Then, you can run lm1b model with data in `<data_dir>` in parallel by executing: 
```shell
$ python lm1b_distributed_driver.py --datadir <data_dir>
```

The command above runs a single LM model on multiple devices specified in `resource_info`.
The command assumes that the data directory and the LM-1B codebase are distributed and reachable in the same absolute path in each of the machines.

Also, we have a few more options you can choose for distributed running.

| Parameter Name       |  Default            	| Description |
| :------------------- |:-----------------------| :-----------|
| --logdir			   | /tmp/lm1b				| Logging directory |
| --datadir			   | None					| Data directory |
| --hpconfig		   | ""						| Overrides default hyper-parameters |
| --eval_steps		   | 70						| Number of evaluation steps |
| --resource_info_file | `./resource_info`		| Filename containing cluster information written |
| --max_steps 		   | 1000000    		    | Number of iterations to run for each workers |
| --log_frequency 	   | 100  		    		| How many steps between two runop log |
| --sync          	   | True  	 				| Whether to synchronize learning or not |
| --ckpt_dir           | None					| Directory to save checkpoints |
| --save_ckpt_steps    | 0						| Number of steps between two consecutive checkpoints |
| --save_n_ckpts_per_epoch | -1					| Number of checkpoints to save per each epoch |
| --run_option		   | None					| The run option whether PS or MPI, None utilizes both |

You can adapt the distributed running with above options. For example, if you want to fix the communication model as MPI mode, you can add `run_option` value like below.

```shell
$ python lm1b_distributed_driver.py --datadir <data_dir> --run_option=MPI
```

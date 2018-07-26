# TensorFlow CNN Benchmarks
The original code of this example comes from [tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks).
We modified this code to build a computation graph for a single-gpu environment instead of a multi-GPU and multi-machine environment(We removed the unnecessary communication-related files like `varialble_mgr.py`, `variable_mgr_util.py`).
We added `CNNBenchmark_distributed_driver.py` for training and `CNNBenchmark_eval.py` for evaluation.

## Dataset
* Synthetic data or imagenet data can be used. To use imagenet data follow these [instructions](https://github.com/tensorflow/models/tree/master/research/inception#getting-started).

## Training
Set your resource information in the `resource_info` file.

Then, execute:
```shell
$ python CNNBenchmark_distributed_driver.py --model={model} --data_name={data_name} --data_dir={data_dir}
```

The command above runs a single CNN model on multiple devices specified in `resource_info`.
The command assumes that the data directory and the TensorFlow CNN benchmark codebase are distributed and reachable in the same absolute path in each of the machines.

Also, we have a few more options you can choose for distributed running.

| Parameter Name       |  Default               | Description |
| :------------------- |:-----------------------| :-----------|
| --resource_info_file | `./resource_info`      | Filename containing cluster information written |
| --max_steps          | 1000000                | Number of iterations to run for each workers |
| --log_frequency      | 100                    | How many steps between two runop log |
| --sync               | True                   | Whether to synchronize learning or not |
| --ckpt_dir           | None                   | Directory to save checkpoints |
| --save_ckpt_steps    | 0                      | Number of steps between two consecutive checkpoints |
| --run_option         | None                   | The run option whether PS or MPI, None utilizes both |

You can adapt the distributed running with above options. For example, if you want to fix the communication model as MPI mode, you can add `run_option` value like below.

```shell
$ python CNNBenchmark_distributed_driver.py --model={model} --data_name={data_name} --data_dir={data_dir} --run_option=MPI
```

## Evaluation
Execute:
```shell
$ python CNNBenchmark_eval.py --eval=True --model={model} --data_name={data_name} --data_dir={data_dir} --checkpoint_dir={checkpoint_dir}
```

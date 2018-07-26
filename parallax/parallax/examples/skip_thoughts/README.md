# Skip-Thought Vectors
This example implements the model described in [Skip-Thought Vectors](https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf). 
The original code comes from [here](https://github.com/tensorflow/models/tree/master/research/skip_thoughts).
We changed a minimal amount of the original code;`import path` code and BUILD file.
We added the `skip_distributed_driver.py` file and modified `ops/input_ops.py`(for data sharding) file to run the example on parallax.

## Dataset
* Follow the instructions shown in [Prepare the Training Data](https://github.com/tensorflow/models/tree/master/research/skip_thoughts).

## To Run
Set your resource information in the `resource_info` file.

Then execute:
```shell
$ python skip_distributed_driver.py --input_file_pattern ${DATA_DIR}/data/train-?????-of-00100
```
The command above runs a single Skip-Thought Vectors model on multiple devices specified in `resource_info`.
The command assumes that the data directory and the Skip-Thought Vectors codebase are distributed and reachable in the same absolute path in each of the machines.

Also, we have a few more options you can choose for distributed running.

| Parameter Name       |  Default               | Description |
| :------------------- |:-----------------------| :-----------|
| --data_path          | None	                | Where to training/test data is stored |
| --input_file_pattern | ""                   	| File pattern of training data |
| --batch_size         | 128                    | Batch size |
| --resource_info_file | `./resource_info`      | Filename containing cluster information written |
| --max_steps          | 1000000                | Number of iterations to run for each workers |
| --log_frequency      | 100                    | How many steps between two runop log |
| --sync               | True                   | Whether to synchronize learning or not |
| --ckpt_dir           | None                   | Directory to save checkpoints |
| --save_ckpt_steps    | 0                      | Number of steps between two consecutive checkpoints |
| --run_option         | None                   | The run option whether PS or MPI, None utilizes both |


You can adapt the distributed running with above options. For example, if you want to fix the communication model as MPI mode, you can add `run_option` value like below.

```shell
$ python skip_distributed_driver.py --input_file_pattern ${DATA_DIR}/data/train-?????-of-00100 --run_option MPI
```

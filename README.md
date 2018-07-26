# Parallax
**Parallax** is a tool that automatically parallelizes training of a single-GPU deep learning model correctly and efficiently in distributed multi-GPU environments. Parallax correctly handles complicated auto-parallelization issues; in addition, it also leverages various optimizations to minimize communication overhead incurred by distributed training.

Parallax is currently implemented on [TensorFlow v1.6](https://github.com/tensorflow/tensorflow/tree/r1.6). In case that Parallax uses Message Passing Interface (MPI), Parallax requires *AllReduce*, *AllGather* operations implemented in [Horovod v0.11.2](https://github.com/uber/horovod/tree/v0.11.2). We plan to support multiple TensorFlow versions.

* [Installation](doc/installation.md)
* [Running Parallax](doc/quick_start.md)
* [Parallax API](doc/parallax_api.md)

## Why Parallax?

Parallax makes it easier for users to do distributed training of a deep learning model developed in a single device (e.g., GPU or CPU). A Parallax user simply specifies a single-device model graph, resource specification for distributed training and Parallax does the rest! For distributed training, Parallax supports two major communication styles: Parameter Server (PS) and Message Passing Interface (MPI). Users can choose which communication method to train their models, or Parallax can choose a communication method that works better automatically.

### Parallax Execution Model

<p align=center><img src=doc/figure/exec_model.png></p>


When a client initiates a deep learning job with a single-device computation graph, resource information, and optionally a flag that indicates either synchronous or asynchronous training, Parallax transforms the computation graph by analyzing its characteristics. Then, Parallax executes the transformed graph with its optimized communication layer in the distributed environment.

### Parallax Benchmark

To give you an idea on how well Parallax performs, we present the following chart that shows the result of experiments done in a cluster of eight machines that are connected via Mellanox ConnectX-4 cards with 100Gbps InfiniBand. Each machine has six NVIDIA GeForce TITAN Xp GPU cards.

<p align=center>
  <img src=/doc/figure/benchmark.png>
</p>
Parallax outperforms TensorFlow for both Resnet50 and LM1B. In addition, Parallax outperforms Horovod for LM1B.

## Troubleshooting
See the [Troubleshooting](doc/trouble_shooting.md) page and submit a new [issue](https://github.com/snuspl/parallax/issues/new) or [contact us](#contact-us) if you cannot find an answer.

## Contact us
To contact us, send an email to parallax-dev@googlegroups.com.

## License
[Apache License 2.0](LICENSE)

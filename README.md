# Parallax
**Parallax** is a tool that optimizes data parallel training by considering whether each variable in a deep learning model is sparse or dense. The sparsity-aware data parallel training improves performance of models with sparse variables that show relatively low scalability on existing frameworks while maintaining equal performance for models with only dense variables such as ResNet-50 and Inception-V3. In addition, Parallax automatically parallelizes training of a single-GPU deep learning model to minimize user efforts. If you are interested, you can find the technical details of Parallax in [our paper](https://dl.acm.org/citation.cfm?id=3303957).

Parallax is currently implemented on TensorFlow. We support [TensorFlow v1.6](https://github.com/tensorflow/tensorflow/tree/r1.6) and [TensorFlow v1.11](https://github.com/tensorflow/tensorflow/tree/r1.11). In case that Parallax uses Message Passing Interface (MPI), Parallax requires *AllReduce*, *AllGather* operations implemented in [Horovod v0.11.2](https://github.com/uber/horovod/tree/v0.11.2). We plan to support multiple TensorFlow versions. 

* [Installation](doc/installation.md)
* [Running Parallax](doc/quick_start.md)
* [Parallax API](doc/parallax_api.md)

## Why Parallax?
Parallax makes it easier for users to do distributed training of a deep learning model developed in a single device (e.g., GPU or CPU) while employing various optimization techniques that Parallax provides. A Parallax user simply specifies a single-device model graph, resource specification for distributed training and Parallax does the rest! For distributed training, Parallax supports hybrid architecture that combines two different distributed training architectures: Parameter Server (PS) and AllReduce (AR). Hybrid architecture exploits the advantages of both architectures. Moreover, Parallax will provide large sparse variable partitioning soon to maximize parallelism while maintaining low computation and communication overhead. Parallax further optimizes training with local aggregation and smart operation placement to mitigate communication overhead. 

PS and AR architectures are still available in Parallax; users can choose the training architecture if they want (default is hybrid for synchronous training).

### Hybrid Architecture
<p align=center><img src=doc/figure/hybrid.png width="400"></p>

The amount of data transfer of each PS and AR achitecture changes according to whether a variable is sparse or dense. Based on the fact, Parallax pursues a hybrid architecture in which the AR architecture handles dense variables and the PS architecture handles sparse variables to minimize communication overhead. Each worker has a replica of dense variables, while separate server processes manage only sparse variables.

### Parallax Execution Model

<p align=center><img src=doc/figure/exec_model.png></p>


When a client initiates a deep learning job with a single-device computation graph, resource information, and optionally a flag that indicates either synchronous or asynchronous training, Parallax transforms the computation graph by analyzing its characteristics. Then, Parallax executes the transformed graph with its optimized communication layer in the distributed environment.

### Parallax Benchmark

To give you an idea on how well Parallax performs, we present the following chart that shows the result of experiments done in a cluster of eight machines that are connected via Mellanox ConnectX-4 cards with 100Gbps InfiniBand. Each machine has six NVIDIA GeForce TITAN Xp GPU cards.

<p float="left">
  <img src="doc/figure/resnet50_convergence.png" width="400" title="ResNet-50"/>
  <img src="doc/figure/lm1b_convergence.png" width="400" title="LM1B"/>
</p>

Parallax converges correctly as other frameworks(TensorFlow and Horovod). Parallax is faster than TensorFlow and similiar to Horovod for ResNet50 (dense model). In case of LM1B (sparse model), Parallax outperforms than both TensorFlow and Horovod.

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

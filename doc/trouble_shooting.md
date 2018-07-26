# Trouble Shooting

Because Parallax execution involves many dependent software and hardware packages, debugging can be tricky if errors occur.
This page collects the troublesome situations we have experienced and the solutions. If you have a similar symptom, try following the suggestions. Also, if you have any additional trouble shooting case, please add it here.

### Device placement error
Error message: 

`device placement error(Cannot assign a device for operation)`

Parallax assumes `allow_soft_placement=True` because Parallax assigns operators on CPU/GPU devices according to their characteristics(shared or replicated) if the placement of the device is not specified. If you face a device placement error, try setting `allow_soft_placement=True` on the session configuration.

### RDMA queue issue while running parameter server model
Error message: 
```
tensorflow/contrib/verbs/rdma.cc:1009] Check failed: status.ok() RecvLocalAsync was not ok. error message: Step 123330693738664103
tensorflow/contrib/verbs/rdma.cc:1009] Check failed: status.ok() RecvLocalAsync was not ok. error message: Step 95609778068110326
```
There are some issues related to managing RDMA queue in Tensorflow. Consider increasing the RDMA queue depth by adjusting `RDMA_QUEUE_DEPTH=<desired_queue_depth>` in `.ssh/environment` or elsewhere you managing environment variables.

### NCCL different version issue
Error message:
```
Signal: Segmentation fault (11)
Signal code: Address not mapped (1)
Failing at address: 0xa0
```
This error can occur if multiple machines use different versions of NCCL. 

### Hang by fetching gradients from non-chief workers while running parameter server model
Error message: None

There are a chief(worker 0) worker and non-chief workers, and Parallax assumes that only the chief worker 
can fetch the gradients. It means fetching gradients from non-chief workers can block the distributed training. 

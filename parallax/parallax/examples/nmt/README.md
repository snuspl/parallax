# Neural Machine Translation (seq2seq)

Neural Machine Translation (NMT) mimics translation process of human. For more detailed description about the program itself, please check out [https://github.com/tensorflow/nmt](https://github.com/tensorflow/nmt) where this program comes from.

## Dataset

We can use the following publicly available datasets:

1. *Small-scale*: English-Vietnamese parallel corpus of TED talks (133K sentence
   pairs) provided by
   the
   [IWSLT Evaluation Campaign](https://sites.google.com/site/iwsltevaluation2015/).
1. *Large-scale*: German-English parallel corpus (4.5M sentence pairs) provided
   by the [WMT Evaluation Campaign](http://www.statmt.org/wmt16/translation-task.html).
   
## To Run

Set your resource information in the `resource_info` file.

The command below runs a single GNMT WMT German-English model on multiple devices specified in `resource_info`. The command assumes that the data directory and the NMT codebase are distributed and reachable in the same absolute path in each of the machines.


```
$ python nmt_distributed_driver.py \ 
    --src=de --tgt=en \
    --hparams_path=nmt/standard_hparams/wmt16_gnmt_4_layer.json \
    --out_dir=/tmp/deen_gnmt \
    --vocab_prefix=/tmp/wmt16/vocab.bpe.32000 \
    --train_prefix=/tmp/wmt16/train.tok.clean.bpe.32000 \
    --dev_prefix=/tmp/wmt16/newstest2013.tok.bpe.32000 \
    --test_prefix=/tmp/wmt16/newstest2015.tok.bpe.32000
```

For more options of nmt model command, please check out [https://github.com/tensorflow/nmt](https://github.com/tensorflow/nmt) again.

Besides, we have a few more options you can choose for distributed running.

| Parameter Name       |  Default            	| Description |
| :------------------- |:-----------------------| :-----------|
| --resource_info_file | `./resource_info`	    | Filename containing cluster information written |
| --max_steps 		   | 1000000    		    | Number of iterations to run for each workers |
| --steps_per_stats 	   | 100  		    		| How many steps between two runop log |
| --sync          	   | True  	 				| Whether to synchronize learning or not |
| --ckpt_dir           | None                   | Directory to save checkpoints |
| --save_ckpt_steps    | 0						| Number of steps between two consecutive checkpoints |
| --run_option		   | None					| The run option whether PS or MPI, None utilizes both |
| --epoch_size 		   | 0						| total number of data instances |

You can adapt the distributed running with above options. For example, you can run the GNMT WMT German-English model in MPI mode by just adding `--run_option` value to the script like below:

```
$ python nmt_distributed_driver.py \ 
    --src=de --tgt=en \
    --hparams_path=${PWD}/nmt/standard_hparams/wmt16_gnmt_4_layer.json \
    --out_dir=/tmp/deen_gnmt \
    --vocab_prefix=/tmp/wmt16/vocab.bpe.32000 \
    --train_prefix=/tmp/wmt16/train.tok.clean.bpe.32000 \
    --dev_prefix=/tmp/wmt16/newstest2013.tok.bpe.32000 \
    --test_prefix=/tmp/wmt16/newstest2015.tok.bpe.32000
    --run_option=MPI 
```

# Transfer Learning for Text Classification with Tensorflow

Tensorflow implementation of [Semi-supervised Sequence Learning(https://arxiv.org/abs/1511.01432)](https://arxiv.org/abs/1511.01432).
 
Auto-encoder or language model is used as a pre-trained model to initialize LSTM text classification model.

- ***SA-LSTM***: Use auto-encoder as a pre-trained model.
- ***LM-LSTM***: Use language model as a pre-trained model.


## Requirements
- Python 3
- Tensorflow
- pip install -r requirements.txt

## Usage
DBpedia dataset is used for pre-training and training.

### Pre-train auto encoder or language model
```
$ python pre_train.py --model="<MODEL>"
```
(\<Model>: auto_encoder | language_model)

### Train LSTM text classification model
```
$ python train.py --pre_trained="<MODEL>"
```
(\<Model>: none | auto_encoder | language_model)


## Experimental Results
- Orange lines: LSTM
- Blue lines: SA-LSTM
- Red lines: LM-LSTM

### Loss
<img src="https://user-images.githubusercontent.com/6512394/42726945-089634b0-87d8-11e8-8f3e-0e986f1d4b51.PNG">

### Accuracy
<img src="https://user-images.githubusercontent.com/6512394/42726944-073e19e8-87d8-11e8-8a60-788fb109ea11.PNG">


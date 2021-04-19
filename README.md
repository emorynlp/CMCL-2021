# Requirements
`virtualenv` is recommended to install the dependencies.
```sh
$ virtualenv env -p python3.6
$ source env/bin/activate
$ pip install -r requirements.txt

```
Because we use a local version of gluon-nlp, we need to set `PYTHONPATH` using `export PYTHONPATH=~/dmlc/gluon-nlp-modified/src/`.
Note that you need to change the path into your own path.

The scripts below are in the directory `scripts`.

# Emotion Classification
## Model Training
The model is trained with three different initialization. See `train.sh` for more tails.
```sh
$ sh train.sh
```

If the parameters are not changed, we would get outputs as below:
```sh
$ ls model_EDQA_MHP_0609_8_32_cv*
model_EDQA_MHP_0609_8_32_run1_gpu0_128_64_dropout0.0:
model_bert_EDQA_0.params
model_bert_EDQA_1.params
model_bert_EDQA_2.params
EDQA_none.tsv

model_EDQA_MHP_0609_8_32_run2_gpu0_128_64_dropout0.0:
model_bert_EDQA_0.params
model_bert_EDQA_1.params
model_bert_EDQA_2.params
EDQA_none.tsv

model_EDQA_MHP_0609_8_32_run3_gpu0_128_64_dropout0.0:
model_bert_EDQA_0.params
model_bert_EDQA_1.params
model_bert_EDQA_2.params
EDQA_none.tsv
```

## Evaluation
Search in the log file to find the checkpoint that gets best results on dev set. For example:
```sh
$ cat nohup.out | grep "Best"
INFO:root:Best model at epoch 2. Validation metrics:accuracy:0.5794
INFO:root:Best model at epoch 2. Validation metrics:accuracy:0.5856
INFO:root:Best model at epoch 2. Validation metrics:accuracy:0.5812
```

Run `test.sh` to predict on test set:
```sh
$ sh test.sh model_EDQA_MHP_0609_8_32_run1_gpu0_128_64_dropout0.0/model_bert_EDQA_2.params output_run1
$ sh test.sh model_EDQA_MHP_0609_8_32_run2_gpu0_128_64_dropout0.0/model_bert_EDQA_2.params output_run2
$ sh test.sh model_EDQA_MHP_0609_8_32_run3_gpu0_128_64_dropout0.0/model_bert_EDQA_2.params output_run3
```

Run `eval.py` to evaluate the predicted results:
```sh
$ python eval.py data/test.bert.csv output_run1/EDQA_none.tsv
$ python eval.py data/test.bert.csv output_run2/EDQA_none.tsv
$ python eval.py data/test.bert.csv output_run3/EDQA_none.tsv
```

# Analysis
## Extract Embeddings
For layer-wise analysis, we need the embeddings of each hidden layer of documents on the trianning set and the dev set.
First check that the parameter `--output_emb` in `test.sh` is set as `True`.
Then run `extract_emb.sh` as below:
```sh
$ sh extract_emb.sh train model_EDQA_MHP_0609_8_32_run1_gpu0_128_64_dropout0.0/model_bert_EDQA_2.params emb_run1_train
$ sh extract_emb.sh valid model_EDQA_MHP_0609_8_32_run1_gpu0_128_64_dropout0.0/model_bert_EDQA_2.params emb_run1_valid
```

In the output directory, there are four files storing the concatenated embeddings of three hidden layers 
and the normalization layer. For example:
```sh
$ ls emb_run1_train/
EDQA_none.norm.emb.tsv # the normalization layer
EDQA_none.128.emb.tsv  # the first hidden layer
EDQA_none.64.emb.tsv   # the second hidden layer
EDQA_none.32.emb.tsv   # the third hidden layer
```

## Classification on Extracted Embeddings 
We train a logistic regression model on the concatenated vector of each layer. Run:
```
sh batch_linear_cls.sh emb_run1
```
The predictions of the dev set are stored in seperated files in the directory `emb_run1_valid` as
`cls_res.32.emb.tsv`, `cls_res.64.emb.tsv` and `cls_res.128.emb.tsv`.


## Generate Emotion Vectors
To generate emotion vectors used in emotion wheel generation and PAD model argumentation, run `get_emb_avg.py` as below:
```sh
$ python get_emb_avg.py emb_run1_valid/EDQA_none.norm.emb.tsv
``` 
The output file will be in the same directory `emb_run1_valid`.


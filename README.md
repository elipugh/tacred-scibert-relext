Position-aware Attention RNN Model for Relation Extraction
=========================

This repo contains the implementation of our [paper](https://github.com/crw2998/tacred-relation-copy/blob/master/papers/final_paper.pdf).


The original code was cloned from from [this repo](https://github.com/yuhaozhang/tacred-relation), which is a *PyTorch* implementation for paper [Position-aware Attention and Supervised Data Improve Slot Filling](https://nlp.stanford.edu/pubs/zhang2017tacred.pdf).

**The TACRED dataset**: Details on the TAC Relation Extraction Dataset can be found on [this dataset website](https://nlp.stanford.edu/projects/tacred/).

## Requirements

- Python 3 (tested on 3.6.2)
- [PyTorch](https://github.com/pytorch/pytorch) (tested on 1.0.0)
- [tqdm](https://github.com/tqdm/tqdm), [bert_as_service](https://github.com/hanxiao/bert-as-service), maybe a couple others
- unzip, wget (for downloading only)

## Preparation

First, download and unzip GloVe vectors from the Stanford website, with:
```
chmod +x download.sh; ./download.sh
```

Then tokenize data to run with BERT or SciBERT with:
```
python data/data_tok.py
```


## Training

Train using the commands in  `cmdcheat.txt`. You need two terminal windows open, or two separate `tmux` sessions. Run the corresponding `bert-as-service` command and then run the `python train.py` command, both with the appropriate flags listed in `cmdcheat.txt`.

Model checkpoints and logs will be saved to `./saved_models/00`.


## Evaluation

Run evaluation on the test set with:
```
python eval.py saved_models/00 --dataset test
```

This will use the `best_model.pt` by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file. Add `--out saved_models/out/test1.pkl` to write model probability output to files (for ensemble, etc.).  

You will need `bert-as-service` running for the test phase as well.

## Ensemble

Please see the example script `ensemble.sh`.

## License

All work contained in this package is licensed under the Apache License, Version 2.0. See the included LICENSE file.

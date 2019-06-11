#!/bin/bash

cd dataset;

echo "==> Downloading BERT..."
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip

echo "==> Unzipping BERT..."
unzip cased_L-12_H-768_A-12.zip
rm cased_L-12_H-768_A-12.zip


echo "==> Downloading SciBERT..."
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_cased.tar.gz

echo "==> Unzipping SciBERT..."
tar -xvzf scibert_scivocab_cased.tar.gz
rm scibert_scivocab_cased.tar.gz

echo "==> Done."



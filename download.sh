#!/bin/bash

cd dataset; mkdir bert
cd bert

echo "==> Downloading glove vectors..."
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip

echo "==> Unzipping glove vectors..."
unzip cased_L-12_H-768_A-12.zip
rm cased_L-12_H-768_A-12.zip

echo "==> Done."



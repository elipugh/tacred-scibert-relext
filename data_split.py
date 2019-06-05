import os
import tokenization
import numpy as np
import json
from random import shuffle

kDataDir = "dataset/life/"
kDataFile = "examples.json"
ksuffix = "scibert"


def split( data_dir, data_file ):

    with open( datafile ) as datafile:
        data = json.load( datafile )

    n = len(data)
    shuffle(data)

    train_data = data[ : int(n*.6) ]
    dev_data = data[ int(n*.6) : int(n*.8) ]
    test_data = data[ int(n*.8) : ]

    with open( kDataDir + "train_"+suffix, 'w+' ) as outfile:
        json.dump( train_data, outfile, indent=2 )

    with open( kDataDir + "dev_"+suffix, 'w+' ) as outfile:
        json.dump( dev_data, outfile, indent=2 )

    with open( kDataDir + "test_"+suffix, 'w+' ) as outfile:
        json.dump( test_data, outfile, indent=2 )



if __name__ == "__main__":
    split( kDataDir, kDataFile, suffix )
import os
import tokenization
import numpy as np
import json
from random import shuffle

kDataDir = "dataset/life/"
kDataFile = "examples.json"
kSuffix = ".json"


def split( ):

    with open( kDataDir+kDataFile ) as datafile:
        data = json.load( datafile )

    n = len(data)

    train_data = data[ : int(n*.6) ]
    dev_data = data[ int(n*.6) : int(n*.8) ]
    test_data = data[ int(n*.8) : ]

    shuffle(train_data)
    shuffle(test_data)
    shuffle(dev_data)

    with open( kDataDir + "train"+kSuffix, 'w+' ) as outfile:
        json.dump( train_data, outfile, indent=2 )

    with open( kDataDir + "dev"+kSuffix, 'w+' ) as outfile:
        json.dump( dev_data, outfile, indent=2 )

    with open( kDataDir + "test"+kSuffix, 'w+' ) as outfile:
        json.dump( test_data, outfile, indent=2 )



if __name__ == "__main__":
    split( )

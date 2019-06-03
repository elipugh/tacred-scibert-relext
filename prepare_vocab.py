"""
Prepare vocabulary and initial word vectors.
"""
import json
import pickle
import argparse
import numpy as np
from collections import Counter

from utils import vocab, constant, helper

def parse_args():
    # for BERT I changed 'glove' and 'wv' in args to 'emb'
    # for BERT, use:
    #   --emb_dir dataset/bert
    #   --emb_file cased_L-12_H-768_A-12 ??
    #   --emb_dim 768 ??
    # not sure on flags yet... not implemented yet.
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('data_dir', help='TACRED directory.')
    parser.add_argument('vocab_dir', help='Output vocab directory.')
    parser.add_argument('--emb_dir', default='dataset/glove', help='Embedding directory.')
    parser.add_argument('--emb_file', default='glove.840B.300d.txt', help='Embedding file.')
    parser.add_argument('--emb_dim', type=int, default=300, help='Embedding dimension.')
    parser.add_argument('--min_freq', type=int, default=0, help='If > 0, use min_freq as the cutoff.')
    parser.add_argument('--lower', action='store_true', help='If specified, lowercase all words.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # input files
    train_file = args.data_dir + '/train.json'
    dev_file = args.data_dir + '/dev.json'
    test_file = args.data_dir + '/test.json'
    emb_file = args.emb_dir + '/' + args.emb_file
    emb_dim = args.emb_dim

    # output files
    helper.ensure_dir(args.vocab_dir)
    vocab_file = args.vocab_dir + '/vocab.pkl'
    output_file = args.vocab_dir + '/embedding.npy'

    # load files
    print("loading files...")
    train_tokens = load_tokens(train_file)
    dev_tokens = load_tokens(dev_file)
    test_tokens = load_tokens(test_file)
    if args.lower:
        train_tokens, dev_tokens, test_tokens = [[t.lower() for t in tokens] for tokens in\
                (train_tokens, dev_tokens, test_tokens)]

    # load glove
    print("loading glove...")
    emb_vocab = vocab.load_glove_vocab(emb_file, emb_dim)
    print("{} words loaded from emb.".format(len(emb_vocab)))
    
    print("building vocab...")
    v = build_vocab(train_tokens, emb_vocab, args.min_freq)

    print("calculating oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))
    
    print("building embeddings...")
    embedding = vocab.build_embedding(emb_file, v, emb_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    np.save(output_file, embedding)
    print("all done.")

def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        for d in data:
            tokens += d['token']
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens

def build_vocab(tokens, glove_vocab, min_freq):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    v = constant.VOCAB_PREFIX + entity_masks() + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched

def entity_masks():
    """ Get all entity mask tokens as a list. """
    masks = []
    subj_entities = list(constant.SUBJ_NER_TO_ID.keys())[2:]
    obj_entities = list(constant.OBJ_NER_TO_ID.keys())[2:]
    masks += ["SUBJ-" + e for e in subj_entities]
    masks += ["OBJ-" + e for e in obj_entities]
    return masks

if __name__ == '__main__':
    main()



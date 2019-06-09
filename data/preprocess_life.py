import csv
import itertools
from collections import defaultdict
import re
import json
import uuid
import random
from tqdm import tqdm

# output
kDataDir = "../dataset/life/"
kSuffix = "_bert.json"

# input
DATADIR = '../dataset/intelligent-life/'
TAXONOMY_STR = 'taxonomy.txt'
STRUCTURE_STR = 'structure.txt'
PROCESS_STR = 'process.txt'
DOCS = ['selected_textbook_sentences', 'life_biology_sentences']
TERM_STR = 'terms.txt'

SAMPLE_NO_RELATION = .13

def load_data():
    with open(DATADIR + TAXONOMY_STR, 'r') as csvfile:
        rels = [[a.strip() for a in rel] for rel in list(csv.reader(csvfile, delimiter='|'))]

    with open(DATADIR + STRUCTURE_STR, 'r') as csvfile:
        # TODO dropped context
        struct_rels = [[a.strip() for a in rel][2:] for rel in list(csv.reader(csvfile, delimiter='|'))]
        rels.extend(struct_rels)

    with open(DATADIR + PROCESS_STR, 'r') as csvfile:
        # TODO dropped context
        process_rels = [[a.strip() for a in rel][2:] for rel in list(csv.reader(csvfile, delimiter='|'))]
        rels.extend(process_rels)

    # TODO take out rels that are too long (longer than 3 words)
    rels = [(r[1],r[2],r[4]) for r in rels if (len(r[1].split()) <= 3 and len(r[4].split()) <= 3)]

    docs = {}
    for file in DOCS:
        with open(DATADIR + file +'.txt', 'r') as sentfile:
            sents = [s.split()[1:] for s in sentfile.read().splitlines()]
        docs[file] = sents

    vocab = set()
    for rel in rels:
        vocab.add(rel[0])
        vocab.add(rel[2])

    return rels, docs, vocab

def words_in_sent(sent, vocab):
    sent_combo = []
    for i in range(len(sent)):
        if sent[i] in vocab:
            sent_combo.append((i, i))
        if i <= len(sent) - 2 and ' '.join(sent[i:i+2]) in vocab:
            sent_combo.append((i, i+1))
        if i <= len(sent) - 3 and ' '.join(sent[i:i+3]) in vocab:
            sent_combo.append((i, i+2))
    return sent_combo

def find_relation(rels_dict, sub, obj):
    if (sub, obj) in rels_dict:
        return rels_dict[(sub, obj)]
    return None

def label_sentences(rels_dict, docs, vocab):
    num_no_relation = 0
    examples = []
    relations = defaultdict(list)
    for docid, sents in docs.items():
        for sent in tqdm(sents):
            word_idxs = words_in_sent(sent, vocab) # returns tuple of "word" indices (can be up to 3)
            for sub_idx, obj_idx in itertools.permutations(word_idxs, 2):
                rel = find_relation(rels_dict,
                    " ".join(sent[sub_idx[0]:sub_idx[1]+1]),
                    " ".join(sent[obj_idx[0]:obj_idx[1]+1]))
                if rel:
                    example = {
                        'docid': docid,
                        'id': uuid.uuid4().hex,
                        'token': sent,
                        'relation': rel[1],
                        'subj_start': sub_idx[0],
                        'subj_end': sub_idx[1],
                        'obj_start': obj_idx[0],
                        'obj_end': obj_idx[1]
                    }
                    examples.append(example)
                    relations[rel].append(example)
                else:
                    if random.random() < SAMPLE_NO_RELATION:
                        num_no_relation += 1
                        example = {
                            'docid': docid,
                            'id': uuid.uuid4().hex,
                            'token': sent,
                            'relation': 'no_relation',
                            'subj_start': sub_idx[0],
                            'subj_end': sub_idx[1],
                            'obj_start': obj_idx[0],
                            'obj_end': obj_idx[1]
                        }
                        examples.append(example)
    return examples, relations, num_no_relation

def save_to_json(examples):
    #with open(DATADIR + 'examples.json', 'w+') as f:
    with open('examples.json', 'w+') as f:
        json.dump(examples, f, indent=4, sort_keys=True)

def train_dev_test_split(relations_dict, examples, train_frac, dev_frac, test_frac):
    num_has_relation = len([i for i in examples if i["relation"] != "no_relation"])
    train, dev = [], []
    relation_keys = set(relations_dict.keys())

    # first we split those with a relation
    # sample for train
    while len(train) < train_frac * num_has_relation:
        rel = random.choice(tuple(relation_keys))
        train += relations_dict[rel]
        relation_keys.remove(rel)

    # sample for dev
    while len(dev) < dev_frac * num_has_relation:
        rel = random.choice(tuple(relation_keys))
        dev += relations_dict[rel]
        relation_keys.remove(rel)

    # the rest are test
    test = []
    for k in relation_keys:
        test += relations_dict[k]

    no_relation = [i for i in examples if i["relation"] == "no_relation"]
    while len(train) < train_frac * len(examples):
        ex = random.randint(0, len(no_relation)-1)
        train.append(no_relation[ex])
        del no_relation[ex]

    while len(dev) < dev_frac * len(examples):
        ex = random.randint(0, len(no_relation)-1)
        dev.append(no_relation[ex])
        del no_relation[ex]

    test += list(no_relation)
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    with open( kDataDir + "train"+kSuffix, 'w+' ) as outfile:
        print ("writing", len(train), "to", outfile)
        json.dump( train, outfile, indent=2 )

    with open( kDataDir + "dev"+kSuffix, 'w+' ) as outfile:
        print ("writing", len(dev), "to", outfile)
        json.dump( dev, outfile, indent=2 )

    with open( kDataDir + "test"+kSuffix, 'w+' ) as outfile:
        print ("writing", len(test), "to", outfile)
        json.dump( test, outfile, indent=2 )

    return train, dev, test


def main():
    rels, docs, vocab = load_data()
    rels_dict = {(r[0], r[2]) : r for r in rels}
    examples, relations_dict, num_no_relation = label_sentences(rels_dict, docs, vocab)
    print('NUM EXAMPLES', len(examples))
    print('NUM NO RELATIONS', num_no_relation)
    save_to_json(examples)
    # do shuffle and split over relations.
    train, dev, test = train_dev_test_split(relations_dict, examples, .6, .2, .2)
    print ("train:", len(train), "examples.")
    print ("dev:", len(dev), "examples.")
    print ("test:", len(test), "examples.")

if __name__ == '__main__':
    main()




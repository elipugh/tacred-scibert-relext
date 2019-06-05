import csv
import itertools
import re
import json
import uuid
from tqdm import tqdm

DATADIR = '../dataset/intelligent-life/'
TAXONOMY_STR = 'taxonomy.txt'
STRUCTURE_STR = 'structure.txt'
PROCESS_STR = 'process.txt'
DOCS = ['selected_textbook_sentences', 'life_biology_sentences']
TERM_STR = 'terms.txt'

SAMPLE_NO_RELATION = .1

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
        docs[file] = label_sentences

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

def find_relation(rels, sub, obj):
    for r in rels:
        if r[0] == sub and r[2] == obj:
            return r
    return None

def label_sentences(rels, docs, vocab):
    examples = []
    for docid, sents in docs.items():
        for sent in tqdm(sents[:500]):
            word_idxs = words_in_sent(sent, vocab) # returns tuple of "word" indices (can be up to 3)
            for sub_idx, obj_idx in itertools.permutations(word_idxs, 2):
                rel = find_relation(rels,
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
                else:
                    if random.random() < SAMPLE_NO_RELATION:
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
    return examples

def save_to_json(examples):
    with open(DATADIR + 'examples.json', 'w') as f:
        json.dump(examples, f, indent=4, sort_keys=True)

def main():
    rels, docs, vocab = load_data()
    examples = label_sentences(rels, docs, vocab)
    print('NUM EXAMPLES', len(examples))
    save_to_json(examples)

if __name__ == '__main__':
    main()




import numpy as np
import json
from collections import Counter
import matplotlib.pyplot as plt

DATASET_DIR = '../dataset/intelligent-life/'


with open(DATASET_DIR + 'examples.json') as f:
    examples = json.load(f)

def plot_counts(data):
    counts = Counter(data)
    del counts["no_relation"]

    labels, values = zip(*counts.items())
    indexes = np.arange(len(labels))
    width = 1

    idx = list(reversed(np.argsort(values)))
    indexes_sorted = indexes[idx]
    values_sorted = np.array(values)[idx]
    labels_sorted = np.array(labels)[idx]
    print(values_sorted)

    plt.bar(range(len(indexes_sorted)), values_sorted, width)
    plt.xticks(indexes_sorted + width * 0.5, labels_sorted, rotation='vertical')
    plt.ylabel("Number of examples")
    plt.tight_layout()
    plt.show()

# relation distribution
print('NUM EXAMPLES', len(examples))
relations = [e['relation'] for e in examples]
print("NUM_UNIQUE_RELATIONS", len(Counter(relations)))
plot_counts(relations)

def plot_counts_sent(data):
    plt.hist(sents, range=(0, 100), bins=100)
    plt.ylabel("Number of examples")
    plt.xlabel("Sentence Length")
    plt.show()

# sentence length distribution
sents = [len(e['token']) for e in examples]
plot_counts_sent(sents)



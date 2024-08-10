import sys
import json

import numpy as np
sys.path.append('../eval')

from edit_distance import find_edit_distance
from compute_ld import compute_lexical_diversity

train_filtered = '../classification/data/hindi-english/train_filtered.tsv'
unfunned = '../hindi_unfuns/train_unfuns.json' #'../hindi_unfuns/filter/train_unfuns.json'
#'../hindi_unfuns/train_unfuns.json'

# train_classifier_filtered = '../hindi_unfuns/train_unfuns_both_reformatted.tsv_filtered_classifier.tsv'
# with open(train_classifier_filtered, 'r') as f:
#     train_classifier_filtered = [x.strip().split('\t') for x in f.readlines()]
#     humorous_classifier_filtered = [x[1] for x in train_classifier_filtered if int(x[2]) == 1]
#     non_humorous_classifier_filtered = [x[1] for x in train_classifier_filtered if int(x[2]) == 0]

with open(train_filtered, 'r') as f:
    train_data = [x.strip().split('\t') for x in f.readlines()]

humorous_data = [x[1] for x in train_data if int(x[2]) == 1]
non_humorous_data = [x[1] for x in train_data if int(x[2]) == 0]


with open(unfunned, 'r') as f:
    unfun_pairs = json.load(f)

edit_distances = [find_edit_distance(x['humor'], x['non_humor']) for x in unfun_pairs]
unfun_data = [x['non_humor'] for x in unfun_pairs]

print('edit distance for unfun:', round(np.mean(edit_distances), 1))
print('lexical diversity for unfun:', round(compute_lexical_diversity(unfun_data)['ttr'], 3))

print()
print('lexical diversity for nonhumor:', round(compute_lexical_diversity(non_humorous_data)['ttr'], 3))
print('lexical diversity for humor:', round(compute_lexical_diversity(humorous_data)['ttr'], 3))
print('lexical diversity for nonhumor + unfun:', round(compute_lexical_diversity(non_humorous_data+unfun_data)['ttr'], 3))
print()
# print('edit distance for filtered nonhumor:', round(np.mean([find_edit_distance(x, y) for x, y in zip(humorous_classifier_filtered, non_humorous_classifier_filtered)]), 1))
# print('lexical diversity for filtered nonhumor:', round(compute_lexical_diversity(non_humorous_classifier_filtered)['ttr'], 3))
# print('lexical diversity for filtered humor:', round(compute_lexical_diversity(humorous_classifier_filtered)['ttr'], 3))

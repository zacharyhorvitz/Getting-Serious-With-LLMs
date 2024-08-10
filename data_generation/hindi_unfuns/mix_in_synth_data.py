path_to_synth_data = 'filter/train_unfuns_just_unfun_reformatted.tsv'
path_to_original_data = '../classification/data/hindi-english/train_filtered.tsv'
percent = 0.50 #25

import random


with open(path_to_original_data, 'r') as f:
    original_data = list(f.readlines())
    positive_data = [line for line in original_data if line.split('\t')[-1].strip() == '1']
    negative_data = [line for line in original_data if line.split('\t')[-1].strip() == '0']
    assert len(positive_data) + len(negative_data) == len(original_data)

with open(path_to_synth_data, 'r') as f:
    synth_data = list(f.readlines())

num_synth = int(len(negative_data) * percent)

random.shuffle(synth_data)
random.shuffle(negative_data)

all_data = synth_data[:num_synth] + negative_data[num_synth:] + positive_data

assert len(all_data) == len(original_data) 

with open(f'filter/train_unfuns_just_unfun_reformatted_{percent}.tsv', 'w') as f:
    for line in all_data:
        f.write(line)


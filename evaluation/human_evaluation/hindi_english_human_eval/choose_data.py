
# for unfun --> satire, we should use high quality unfun (†est)
# for satire --> unfun, we can use any human data


# For all models, we need to sample 100 examples for the headlines we choose
# and then generate them with the model.

import sys
import random
import os
import re
import json

sys.path.append('../utils/')

from text_utils import load_tsv_data

def clean(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'www\S+', '', tweet)

    # Remove HTML tags
    tweet = re.sub(r'<.*?>', '', tweet)
    
    return tweet


if __name__ == '__main__':

    random.seed(42)
    ORIGINAL_TEST_DATA = 'test.tsv' # humorous and non humorous
    UNFUN_TEST_DATA = 'test_unfuns.json' # unfunny and funny
    NUM_SAMPLES = 125
    EXAMPLES = 5


    # original test data isnt filtered yet
    test_data = load_tsv_data(ORIGINAL_TEST_DATA)

    nonfunny_test_data = [x for x in test_data if int(x[2]) == 0]
    nonfunny_test_data = [x for x in nonfunny_test_data if ".twitter."  not in x[1]]
    nonfunny_test_data = [(x[0], clean(x[1])) for x in nonfunny_test_data]

    with open(UNFUN_TEST_DATA, 'r') as f:
        unfun_pairs_test_data = json.load(f)

    nonfun_indices = list(range(len(nonfunny_test_data)))
    random.shuffle(nonfun_indices)

    nonfunny_indices = nonfun_indices[:NUM_SAMPLES]
    selected_nonfunny_test_data = [(i, nonfunny_test_data[i]) for i in nonfunny_indices]

    unfun_pairs_indices = list(range(len(unfun_pairs_test_data)))
    random.shuffle(unfun_pairs_indices)

    humor_indices = unfun_pairs_indices[:NUM_SAMPLES]
    unfun_indices = unfun_pairs_indices[NUM_SAMPLES:2*NUM_SAMPLES]

    humor_test_data = [(i, unfun_pairs_test_data[i]) for i in humor_indices]
    unfun_test_data = [(i, unfun_pairs_test_data[i]) for i in unfun_indices]

    # check no overlap
    assert len(set(humor_indices).intersection(set(unfun_indices))) == 0
    assert len(set([json.dumps(x) for x in humor_test_data]).intersection(set([json.dumps(x) for x in unfun_test_data]))) == 0

    with open('humor_eval_data.tsv', 'w') as f:
        for i, x in humor_test_data:
            f.write(f'{i}\t{x["humor"]}\n')

    with open('unfunned_eval_data.tsv', 'w') as f:
        for i, x in unfun_test_data:
            f.write(f'{i}\t{x["non_humor"]}\n')

    with open('nonfunny_eval_data.tsv', 'w') as f:
        for i, x in selected_nonfunny_test_data:
            f.write(f'{i}\t{x[1]}\n')


    # save examples for each
    humor_example_indices = unfun_pairs_indices[2*NUM_SAMPLES:2*NUM_SAMPLES+EXAMPLES]
    humor_examples = [(i, unfun_pairs_test_data[i]) for i in humor_example_indices]

    unfun_example_indices = unfun_pairs_indices[2*NUM_SAMPLES+EXAMPLES:2*NUM_SAMPLES+2*EXAMPLES]
    unfun_examples = [(i, unfun_pairs_test_data[i]) for i in unfun_example_indices]

    nonfunny_example_indices = nonfun_indices[:EXAMPLES]
    nonfunny_examples = [(i, nonfunny_test_data[i]) for i in nonfunny_example_indices]

    # check no overlap
    assert len(set(humor_example_indices).intersection(set(unfun_example_indices))) == 0
    assert len(set(humor_example_indices).intersection(set(humor_indices))) == 0
    assert len(set(unfun_example_indices).intersection(set(unfun_indices))) == 0

    with open('humor_eval_examples.tsv', 'w') as f:
        for i, x in humor_examples:
            f.write(f'{i}\t{x["humor"]}\n')

    with open('unfunned_eval_examples.tsv', 'w') as f:
        for i, x in unfun_examples:
            f.write(f'{i}\t{x["non_humor"]}\n')

    with open('nonfunny_eval_examples.tsv', 'w') as f:
        for i, x in nonfunny_examples:
            f.write(f'{i}\t{x[1]}\n')

    





    



    








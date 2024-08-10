
# for unfun --> satire, we should use high quality unfun (†est)
# for satire --> unfun, we can use any human data


# For all models, we need to sample 100 examples for the headlines we choose
# and then generate them with the model.

import sys
import random
import os

sys.path.append('../utils/')

from text_utils import load_tsv_data

def load_unfun_pair_data(path):
        return load_tsv_data(path)

def sample_no_duplicates(data, sample_size, filter_fn=None, avoid_data=[]):
    seen = set()
    if avoid_data:
        for sample in avoid_data:
            if isinstance(sample, list):
                unfun_id, unfun, satire_id, satire, url = sample
                seen.add(satire.lower())
            elif isinstance(sample, str):
                seen.add(sample.lower())
            else:
                raise ValueError('Unexpected data format: {}'.format(sample))
        
    sampled = []
    while len(sampled) < sample_size:
        unfun_id, unfun, satire_id, satire, url = random.choice(data)
        del url

        unfun = unfun.lower()
        satire = satire.lower()

        if filter_fn is not None and filter_fn(unfun, satire):
            continue

        if satire in seen:
            continue

        seen.add(satire)
        sampled.append((unfun_id, unfun, satire_id, satire))
        
    return sampled

if __name__ == '__main__':

    HIGH_QUALITY_UNFUN = '../classification/data/unfun/paired/test_unique_pairs_no_leakage.tsv'
    SATIRE_TO_TRUE_UNFUN_DISTRIBUTION = '../classification/data/unfun/raw/count_na-prefix_shared_0-serious_na-funny_na-worddiff_0-SameLen_0.tsv'
    PROMPT_DATA_TO_AVOID = '../classification/data/unfun/paired/train_for_prompting.tsv'
    REAL_DATA = '../classification/data/unfun/real_news/unfun_real_headlines.tsv'

    REAL_SAMPLES_TO_AVOID = ["the word 'doofuses' may cost ex-yahoo ceo bartz $10 million", '2 meteorites hit connecticut', "world outraged by north korea's latest nuke test", 'poverty rate hits 17-year high', 'philippines: 5 foreign terror suspects in south']


    SAMPLE_SIZE = 200

    

    random.seed(42)

    avoid_data = load_tsv_data(PROMPT_DATA_TO_AVOID)
    pairs_for_satire = load_unfun_pair_data(HIGH_QUALITY_UNFUN)
    pairs_for_unfun = load_unfun_pair_data(SATIRE_TO_TRUE_UNFUN_DISTRIBUTION)

    real_data = [x+[x[1]]*3 for x in load_tsv_data(REAL_DATA)]

    sample_for_satire = sample_no_duplicates(pairs_for_satire, SAMPLE_SIZE, avoid_data=avoid_data)
    sample_for_unfun = sample_no_duplicates(pairs_for_unfun, SAMPLE_SIZE, filter_fn=lambda unfun, satire: unfun == satire or unfun == 'none', avoid_data=avoid_data)

    sample_for_real = sample_no_duplicates(real_data, SAMPLE_SIZE, avoid_data = REAL_SAMPLES_TO_AVOID)

    print(sample_for_satire[0])
    print(sample_for_unfun[0])
    print(sample_for_real[0])

    dirname = f'sample_{SAMPLE_SIZE}'

    os.makedirs(dirname, exist_ok=False)

    with open(f'{dirname}/pairs_for_satire_gen.tsv', 'w') as f:
        for row in sample_for_satire:
            f.write('\t'.join(row) + '\n')
        
    with open(f'{dirname}/pairs_for_unfun_gen.tsv', 'w') as f:
        for row in sample_for_unfun:
            f.write('\t'.join(row) + '\n')

    with open(f'{dirname}/real_heads.tsv', 'w') as f:
        for row in sample_for_real:
            f.write('\t'.join(row) + '\n')




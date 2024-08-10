
# for unfun --> satire, we should use high quality unfun (†est)
# for satire --> unfun, we can use any human data


# For all models, we need to sample 100 examples for the headlines we choose
# and then generate them with the model.

import sys
import random
import os
import re
import json
import glob

sys.path.append('../../../utils/')

from text_utils import load_tsv_data

def clean(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'www\S+', '', tweet)

    # Remove HTML tags
    tweet = re.sub(r'<.*?>', '', tweet)
    
    return tweet


if __name__ == '__main__':

    for path in glob.glob('*.tsv'):
        if '_filtered.tsv' in path:
            continue
    
        data = load_tsv_data(path)

        data = [x for x in data if ".twitter."  not in x[1]]
        data = [(x[0], clean(x[1]),x[2]) for x in data]

        with open(path.replace('.tsv', '_filtered.tsv'), 'w') as f:
            for row in data:
                f.write('\t'.join(row) + '\n')




    



    








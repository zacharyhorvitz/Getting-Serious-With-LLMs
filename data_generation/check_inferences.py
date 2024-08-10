import glob


paths = 'inferences'


import os
import sys
import glob
sys.path.append('../utils/')

from text_utils import load_tsv_data

def normalize(texts):
    texts = [' '.join(x.lower().split()) for x in texts]
    # replace common unicode characters
    texts = [x.replace('’', "'").replace('‘',"'").replace('“', '"').replace('”', '"').replace('—', '-').replace('…', '...').replace('</s>','') for x in texts]
    
    return texts


unfun_paths = 'inferences/unfun_*/*/model_outputs.tsv'
satire_paths = 'inferences/satire_*/*/model_outputs.tsv'

all_paths = list(glob.glob(unfun_paths)) + list(glob.glob(satire_paths))

def check_equivalent(data):
    total = 0
    identical = 0
    for d in data:
        if normalize([d[-1]]) == normalize([d[-2]]):
            identical += 1
        total += 1

    return identical / total
    


for path in all_paths:
    data = load_tsv_data(path)
    print(path)
    print('\t', check_equivalent(data))




import sys
import glob
import os
import json

import random

random.seed(42)

sys.path.append('../utils/')

from text_utils import load_tsv_data

def get_paths(pattern):
    return list(glob.glob(pattern))

def filter_model_paths(paths, models):
    return [path for path in paths if any(model in path for model in models)]

NUM_ANNOTATORS = 3
START_IDX = 0

EVAL_NAME = 'TEST_ANNOTATIONS_FULL'
MODELS=['gpt4']
DEBUG=True

humor_eval_data = load_tsv_data('humor_eval_data.tsv')
unfun_eval_data = load_tsv_data('unfunned_eval_data.tsv')
nonfun_eval_data = load_tsv_data('nonfunny_eval_data.tsv')


experiment_name = f'{EVAL_NAME}_START_{START_IDX}_{NUM_ANNOTATORS}_{"_".join(MODELS)}'

os.makedirs(f'annotations/{experiment_name}/assignments', exist_ok=False)

columns = ['id', 'tweet', 'humorous/non-humorous {h/n}', 'coherent {0,1}']

annotator_data = []
annotator_data += [(j,'humor',*x) for j,x in enumerate(humor_eval_data)]
annotator_data += [(j,'unfunned',*x) for j,x in enumerate(unfun_eval_data)]
annotator_data += [(j,'nonfunny',*x) for j,x in enumerate(nonfun_eval_data)]

for i in range(NUM_ANNOTATORS):

    random.shuffle(annotator_data)

    with open(f'annotations/{experiment_name}/annotator_{i}.tsv', 'w') as f:
        if DEBUG:
            f.write('\t'.join(['task_type', 'orig_id'] + columns) + '\n')
        else:
            f.write('\t'.join(columns) + '\n')

        for idx, (j,task, orig_id, text) in enumerate(annotator_data):
            if DEBUG:
                f.write(f'{task}\t{j}-{orig_id}\t{idx}\t{text}'+ '\t' * (len(columns) - 3) + '\n')
            else:
                f.write(f'{idx}\t{text}'+ '\t' * (len(columns) - 1) + '\n')


    as_dict = {k: v for k, v in enumerate(annotator_data)}

    with open(f'annotations/{experiment_name}/assignments/annotator_{i}.json', 'w') as f:
        json.dump(as_dict, f, indent=2)

    












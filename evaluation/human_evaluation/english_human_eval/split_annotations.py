
import sys
import glob
import os
import json

import random

random.seed(42)

from choose_data import load_unfun_pair_data



def normalize(texts):
    texts = [' '.join(x.lower().split()) for x in texts]
    # replace common unicode characters
    texts = [x.replace('’', "'").replace('‘',"'").replace('“', '"').replace('”', '"').replace('—', '-').replace('…', '...').replace('</s>','') for x in texts]
    
    return texts

def load_and_normalize(path):
    data = load_unfun_pair_data(path)
    return [normalize(x) for x in data]

def get_paths(pattern):
    return list(glob.glob(pattern))

def filter_model_paths(paths, models):
    return [path for path in paths if any(model in path for model in models)]

ANNOTATIONS_PER_SAMPLE = 3
SATIRE_NUM_MODELS = 3
UNFUN_NUM_MODELS = 4
NUM_SAMPLES = 100
NUM_ANNOTATORS = 10
START_IDX = 20

EVAL_NAME = 'TEST_ANNOTATIONS_FULL'

MODELS=['gpt4', 'chatgpt', 'mistral_instruct', 'unfun_roberta-base']

TOTAL = ANNOTATIONS_PER_SAMPLE * NUM_SAMPLES * (SATIRE_NUM_MODELS + UNFUN_NUM_MODELS + 3) * NUM_ANNOTATORS

TOTAL_PER_ANNOTATOR = TOTAL / NUM_ANNOTATORS
print('Total annotations: {}'.format(TOTAL))
print('Total annotations per annotator: {}'.format(TOTAL_PER_ANNOTATOR))

PAIRS_FOR_SATIRE_GEN = 'sample_200/pairs_for_satire_gen.tsv'
PAIRS_FOR_UNFUN_GEN = 'sample_200/pairs_for_unfun_gen.tsv'

REAL_DATA = 'sample_200/real_heads.tsv'

SATIRE_OUTPUTS = get_paths('inferences_200/satire_*/*/model_outputs.tsv')
UNFUN_OUTPUTS = get_paths('inferences_200/unfun_*/*/model_outputs.tsv')

SATIRE_OUTPUTS = filter_model_paths(SATIRE_OUTPUTS, MODELS)
UNFUN_OUTPUTS = filter_model_paths(UNFUN_OUTPUTS, MODELS)

DEBUG = False # whether to use model paths in annotation files

assert len(SATIRE_OUTPUTS) == SATIRE_NUM_MODELS, SATIRE_OUTPUTS
assert len(UNFUN_OUTPUTS) == UNFUN_NUM_MODELS, UNFUN_OUTPUTS

assert len(set(SATIRE_OUTPUTS).intersection(set(UNFUN_OUTPUTS))) == 0

print('Minimum annotators for unique votes:', ANNOTATIONS_PER_SAMPLE * (SATIRE_NUM_MODELS + UNFUN_NUM_MODELS + 3))

experiment_name = f'{EVAL_NAME}_START_{START_IDX}_{NUM_SAMPLES}_{NUM_ANNOTATORS}_{"_".join(MODELS)}'

os.makedirs(f'annotations/{experiment_name}/assignments', exist_ok=False)

def pair_up_inputs_and_outputs(input_path, output_paths):

    inputs = load_and_normalize(input_path)
    path_to_outputs = {}

    assert len(inputs) >= NUM_SAMPLES

    inputs = inputs[START_IDX:START_IDX+NUM_SAMPLES]

    for output_path in output_paths:
        path_to_outputs[output_path] = load_and_normalize(output_path)
        path_to_outputs[output_path] = path_to_outputs[output_path][START_IDX:START_IDX+NUM_SAMPLES]
        assert len(path_to_outputs[output_path]) == len(inputs)

    paired_results = {}

    for i in range(len(inputs)):
        paired_results[i] = {f'[GOLD]': inputs[i]}
        for output_path in output_paths:
            #   print(inputs[i])
            #   print( path_to_outputs[output_path][i])
              assert path_to_outputs[output_path][i][1] in inputs[i], f'{output_path} at {i} not in {input_path} with {path_to_outputs[output_path][i][1]} and {inputs[i]}'
              paired_results[i][output_path] = path_to_outputs[output_path][i]

    return paired_results

paired_unfuns = pair_up_inputs_and_outputs(PAIRS_FOR_UNFUN_GEN, UNFUN_OUTPUTS)
paired_satires = pair_up_inputs_and_outputs(PAIRS_FOR_SATIRE_GEN, SATIRE_OUTPUTS)
unpaired_real = {f'real_{i}': {REAL_DATA:x[:2]} for i,x in enumerate(load_and_normalize(REAL_DATA)[START_IDX:START_IDX+NUM_SAMPLES])}


def generate_assignents(paired_results):
    annotator_ids = list(range(NUM_ANNOTATORS))
    annotator_to_assignments = {annotator_id: [] for annotator_id in annotator_ids}

    paired_keys = sorted(list(paired_results.keys()))
    model_paths = sorted(list(paired_results[paired_keys[0]].keys()))
    random.shuffle(paired_keys)

    cur_annotator_idx = 0
    for k in paired_keys:
        random.shuffle(model_paths)
        for model_path in model_paths:
            for j in range(ANNOTATIONS_PER_SAMPLE):
                annotator_to_assignments[annotator_ids[cur_annotator_idx]].append((k, model_path, j))
                if cur_annotator_idx == len(annotator_ids) - 1:
                    cur_annotator_idx = 0
                else:
                    cur_annotator_idx += 1


    # check that no annotator has more than one assignment for a given output
    for annotator_id in annotator_to_assignments:
        seen = [(k,model_path) for k, model_path, _ in annotator_to_assignments[annotator_id]]
        assert len(seen) == len(set(seen))
           
    return annotator_to_assignments


annotator_unfun_assignments = generate_assignents(paired_unfuns)
annotator_satire_assignments = generate_assignents(paired_satires)
annotator_real_assignments = generate_assignents(unpaired_real)

combined_assignments = {}
for annotator_id in annotator_unfun_assignments:
    satire = annotator_satire_assignments[annotator_id]
    unfun = annotator_unfun_assignments[annotator_id]
    real = annotator_real_assignments[annotator_id]
    combined_assignments[annotator_id] = list(zip(['satire'] * len(satire), satire)) + list(zip(['unfun'] * len(unfun), unfun)) + list(zip(['real'] * len(real), real))
    random.shuffle(combined_assignments[annotator_id])


# Save assignments
for annotator_id in combined_assignments:
    print('Annotator {}: {}'.format(annotator_id, len(combined_assignments[annotator_id])))

    as_dict = {k: v for k, v in enumerate(combined_assignments[annotator_id])}
    with open(f'annotations/{experiment_name}/assignments/annotator_{annotator_id}.json', 'w') as f:
        json.dump(as_dict, f, indent=2)

# Construct the annotation files
        
columns = ['id', 'headline', 'real/satirical/neither {r/s/n}', 'funniness {0,1,2}', 'grammatical {0,1}',  'coherent {0,1}']

for annotator_id in combined_assignments:
    print('Annotator {}: {}'.format(annotator_id, len(combined_assignments[annotator_id])))

    with open(f'annotations/{experiment_name}/annotator_{annotator_id}.tsv', 'w') as f:
        if DEBUG:
            f.write('\t'.join(['task_type', 'model_path'] + columns) + '\n')
        else:
            f.write('\t'.join(columns) + '\n')
        for idx, (task_type, (k, model_path, j)) in enumerate(combined_assignments[annotator_id]):
            if task_type == 'satire':
                data = paired_satires[k][model_path]
            elif task_type == 'unfun':
                data = paired_unfuns[k][model_path]
            elif task_type == 'real':
                data = unpaired_real[k][model_path][:2]
            else:
                raise ValueError('Invalid task type: {}'.format(task_type))
            
            if model_path == '[GOLD]' and task_type == 'satire':
                text = data[3]
            elif model_path == '[GOLD]' and task_type in ['unfun', 'real']:
                text = data[1]
            else:
                text = data[-1]

            text = text.replace('</s>','')

            if DEBUG:
                f.write(f'{task_type}\t{model_path}\t{idx}\t{text}'+ '\t' * (len(columns) - 2) + '\n')
            else:
                f.write(f'{idx}\t{text}'+ '\t' * (len(columns) - 1) + '\n')



            
            

















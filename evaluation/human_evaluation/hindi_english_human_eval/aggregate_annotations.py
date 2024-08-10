import os
import pandas as pd
import json
import numpy as np
from collections import Counter
INPUT_DIRECTORY = 'annotations/TEST_ANNOTATIONS_FULL_START_0_3_gpt4'

NUM_SAMPLE_ANNOTATIONS = 3
allow_missing = False

FILTERED_TEST_DATA_PATH = '../hindi_unfuns/filter/test_unfuns.json'
with open(FILTERED_TEST_DATA_PATH, 'r') as f:
    FILTERED_UNFUNS = set([x['non_humor'] for x in json.load(f)]) # if x['non_humor'] != x['humor']])
    # FILTERED_UNFUNS = set([x['non_humor'] for x in json.load(f) if x['non_humor'] == x['humor']])

def mean_of_mean(labels):
    assert len(labels) == NUM_SAMPLE_ANNOTATIONS
    return np.mean([np.mean(x) for x in labels])

def mean_of_mode(labels):
    return np.mean([Counter(x).most_common(1)[0][0] for x in labels])

def normalize(texts):
    texts = [' '.join(str(x).lower().split()) for x in texts]
    # replace common unicode characters
    texts = [x.replace('’', "'").replace('‘',"'").replace('“', '"').replace('”', '"').replace('—', '-').replace('…', '...').replace('</s>','') for x in texts]
    
    return texts

def check_consistency(template_file, result_file, n=4):
    template_data = pd.read_csv(template_file, sep='\t')
    result_data = pd.read_csv(result_file, sep='\t')

    # check if headline column is the same after normalization


    template_headlines = normalize(template_data['tweet'].tolist()) # because of issue with the template file
    result_headlines = normalize(result_data['tweet'].tolist())
   
    for i in range(n):
        assert template_headlines[i] == result_headlines[i], f'tweet {i} is not the same: {template_headlines[i]} vs {result_headlines[i]}'

def format_annotation(annotation):
    errors = []

    # id	tweet	humorous/non-humorous {h/n}	coherent {0,1}

    type_key = 'humorous/non-humorous {h/n}'
    coherent_key = 'coherent {0,1}'
    valid_type_key = ['h', 'n']


    # print(annotation)
    annotation[type_key] = annotation[type_key].lower()
    if annotation[type_key] not in valid_type_key:
        errors.append(f'Invalid type: {annotation[type_key]}')

    assert annotation[coherent_key] in [0, 1], f'Invalid coherent: {annotation[coherent_key]}'

    annotation['is humorous'] = 1 if annotation[type_key] == 'h' else 0
    annotation['is nonhumorous'] = 1 if annotation[type_key] == 'n' else 0

    # annotation['gpt4_labeled'] = 1  if annotation['tweet'] in FILTERED_UNFUNS else 0
    

    return errors

def get_labels(file_data, label):
    labels = []
    texts = []
    for annotations in file_data['entries'].values():
        labels.append([])
        texts.append([])
        for annotation in annotations:
            if annotation and annotation[label] != -1:
                labels[-1].append(annotation[label])
                texts[-1].append(annotation['tweet'])


    return labels, texts
    
def get_results(file, file_to_annotations, key_order):
    # print(f'Processing file {file}')
    for entry in file_to_annotations[file]['entries']:
        num = len([x for x in file_to_annotations[file]['entries'][entry] if x])

        if allow_missing and num != NUM_SAMPLE_ANNOTATIONS:
                print(f'WARNING: File {file} entry {entry} only has {num} annotations')
        elif num != NUM_SAMPLE_ANNOTATIONS:
            
            assert False, f'Not all annotators annotated file {file} entry {entry} ({num})'
        
    results = []
    for label in key_order:
        labels, texts = get_labels(file_to_annotations[file], label)

        # check that all texts are the same
        for i in range(len(texts)):
            assert len(set(texts[i])) == 1, f'Not all texts are the same for file {file} label {label}'
            texts[i] = texts[i][0]

        # print(label, labels)
        
        # if 'funniness' in label:
            # aggregate = mean_of_mean(labels)
        # else:
        aggregate = mean_of_mode(labels) #* 100


    
        results.append(aggregate)
        # print(texts)
        gpt4_filtered = [x for x, t in zip(labels, texts) if t in FILTERED_UNFUNS]
        if len(gpt4_filtered) > 0:
            print(len(gpt4_filtered))
            results.append((mean_of_mode(gpt4_filtered),))
    
    return results


results_dir = os.path.join(INPUT_DIRECTORY, 'results')
assert os.path.exists(results_dir)

assignments_dir = os.path.join(INPUT_DIRECTORY, 'assignments')
num_annotators = len(os.listdir(assignments_dir))

annotator_to_data = {}




for i in range(num_annotators):

    generated_template = os.path.join(INPUT_DIRECTORY, f'annotator_{i}.tsv')
    assert os.path.exists(generated_template)

    # assignments

    assignments = os.path.join(assignments_dir, f'annotator_{i}.json') 
    assert os.path.exists(assignments)

    # result_file = os.path.join(results_dir, f'annotator_{i}.tsv')
    result_file = os.path.join(results_dir,f'annotations_{i} - annotations.tsv')

    if allow_missing and not os.path.exists(result_file):
        print(f'WARNING: {i} - annotations.tsv does not exist')
        continue

    assert os.path.exists(result_file), f'{result_file} does not exist'

    check_consistency(generated_template, result_file)

    # read assignments
    with open(assignments, 'r') as f:
        data = json.load(f)
    
    annotator_to_data[i] = data
    
    # read results
    results = pd.read_csv(result_file, sep='\t')

    # replace nan with empty string
    results = results.fillna(-1)

    for k in range(len(results)):
        row_data = results.iloc[k]
        id = str(row_data['id'])
        annotation = row_data.to_dict()

        errors = format_annotation(annotation)
        assert len(errors) == 0, f'Errors in row {result_file} {k}: {errors}'
        annotator_to_data[i][id].append(annotation)


file_to_annotations = {}
# import pdb; pdb.set_trace()
for annotator_id in annotator_to_data:
    for values  in annotator_to_data[annotator_id].values():
        # print(values)

        #  "374": [
    # 79,
    # "nonfunny",
    # "174",
    # "Roz daaru roz daaru roz roz roz roz daaru:"
#   ]
        file = values[1]
        entry = values[2]
        annotation_data = values[4]
        
#         task, (entry, file, annotation_idx), annotation_data = values

#         file = task+'_'+file

        if file not in file_to_annotations:
            file_to_annotations[file] = {'entries': {}}

        if entry not in file_to_annotations[file]['entries']:
            file_to_annotations[file]['entries'][entry] = [[] for _ in range(num_annotators)]
        
        file_to_annotations[file]['entries'][entry][annotator_id] = annotation_data


with open('HINDI_file_to_annotations.json', 'w') as f:
    json.dump(file_to_annotations, f)

for file in file_to_annotations:
    assert len(file_to_annotations[file]['entries']) == 125
    for entry in file_to_annotations[file]['entries']:
        assert len(file_to_annotations[file]['entries'][entry]) == NUM_SAMPLE_ANNOTATIONS

# all_agreements = []

# key_order = [ 'is satirical', 'is real', 'funniness {0,1,2}',  'grammatical {0,1}', 'coherente {0,1}']
key_order = [ 'is humorous', 'coherent {0,1}']
files = sorted(file_to_annotations.keys())

for file in files:
    print(file, get_results(file, file_to_annotations, key_order))


labels, texts = get_labels(file_to_annotations['unfunned'], 'is humorous')
coherent_labels, _ = get_labels(file_to_annotations['unfunned'], 'coherent {0,1}')
unfunned_non_humorous = [(l,l_c, t[0]) for l, l_c, t in zip(labels, coherent_labels, texts) if Counter(l).most_common(1)[0][0] == 0 and Counter(l_c).most_common(1)[0][0] == 1]


# write unfunned_non_humorous to file
with open('unfunned_human_non_humorous.tsv', 'w') as f:
    for i,(_, _, t) in enumerate(unfunned_non_humorous):
        f.write(f'{i}\t{t}\t{0}\n')
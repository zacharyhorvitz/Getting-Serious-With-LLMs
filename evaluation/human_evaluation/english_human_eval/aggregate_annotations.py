import os
import pandas as pd
import json
import numpy as np
from collections import Counter
INPUT_DIRECTORY = 'annotations/TEST_ANNOTATIONS_FULL_START_20_100_10_gpt4_chatgpt_mistral_instruct_unfun_roberta-base'
# 'annotations/TEST_ANNOTATIONS_PILOT_20_2_gpt4_mistral_instruct_unfun_roberta-base'
NUM_SAMPLE_ANNOTATIONS = 3
allow_missing = False

def mean_of_mean(labels):
    return np.mean([np.mean(x) for x in labels])

def mean_of_mode(labels):
    return np.mean([Counter(x).most_common(1)[0][0] for x in labels])

def normalize(texts):
    texts = [' '.join(x.lower().split()) for x in texts]
    # replace common unicode characters
    texts = [x.replace('’', "'").replace('‘',"'").replace('“', '"').replace('”', '"').replace('—', '-').replace('…', '...').replace('</s>','') for x in texts]
    
    return texts

def check_consistency(template_file, result_file, n=4):
    template_data = pd.read_csv(template_file, sep='\t')
    result_data = pd.read_csv(result_file, sep='\t')

    # check if headline column is the same after normalization


    template_headlines = normalize(template_data['id'].tolist()) # because of issue with the template file
    result_headlines = normalize(result_data['headline'].tolist())
   
    for i in range(n):
        assert template_headlines[i] == result_headlines[i], f'Headline {i} is not the same: {template_headlines[i]} vs {result_headlines[i]}'

def format_annotation(annotation):
    errors = []

    type_key = 'real/satirical/neither {r/s/n}'
    coherent_key = 'coherent {0,1}'
    grammatical_key = 'grammatical {0,1}'
    funniness_key = 'funniness {0,1,2}'
    valid_type_key = ['r', 's', 'n']

    

    enforced_combo = {
        'r':  {coherent_key: [-1], grammatical_key: [-1], funniness_key: [-1]},
        's':  {coherent_key: [-1], grammatical_key: [-1], funniness_key: [0, 1, 2]},
        'n' :  {coherent_key: [0, 1], grammatical_key:  [0,1], funniness_key: [-1]},
    }
    # print(annotation)
    annotation[type_key] = annotation[type_key].lower()
    if annotation[type_key] not in valid_type_key:
        errors.append(f'Invalid type: {annotation[type_key]}')

    for k in enforced_combo[annotation[type_key]]:
        annotation[k] = int(annotation[k])
        if annotation[k] not in enforced_combo[annotation[type_key]][k]:
            errors.append(f'Invalid value for {k}: {annotation[k]}')

    # set label to coherent and grammatical if type is not neither
    if annotation[type_key] != 'n':
        annotation[coherent_key] = 1
        annotation[grammatical_key] = 1

    if annotation[type_key] in ['r', 'n']:
        annotation[funniness_key] = 0

    annotation['is real'] = 1 if annotation[type_key] == 'r' else 0
    annotation['is satirical'] = 1 if annotation[type_key] == 's' else 0
    annotation['is funny'] = 1 if annotation[funniness_key] > 0 else 0
    annotation['very funny'] = 1 if annotation[funniness_key] > 1 else 0


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
                texts[-1].append(annotation['headline'])


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
        labels, _ = get_labels(file_to_annotations[file], label)
        if 'funniness' in label:
            aggregate = mean_of_mean(labels)
        else:
            aggregate = mean_of_mode(labels) #* 100
        results.append(aggregate)
    
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
    if allow_missing and not os.path.exists(os.path.join(results_dir,f'{i} - annotations.tsv')):
        print(f'WARNING: {i} - annotations.tsv does not exist')
        continue

    result_file = os.path.join(results_dir,f'{i} - annotations.tsv')
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

for annotator_id in annotator_to_data:
    for values  in annotator_to_data[annotator_id].values():
        
        task, (entry, file, annotation_idx), annotation_data = values

        file = task+'_'+file

        if file not in file_to_annotations:
            file_to_annotations[file] = {'entries': {}}

        if entry not in file_to_annotations[file]['entries']:
            file_to_annotations[file]['entries'][entry] = [[] for _ in range(num_annotators)]
        
        file_to_annotations[file]['entries'][entry][annotator_id] = annotation_data
           
with open('HUMAN_EVAL_aggregated_results.json', 'w') as f:
    json.dump(file_to_annotations, f, indent=4)

all_agreements = []

key_order = [ 'is real', 'is funny', 'very funny',  'grammatical {0,1}', 'coherent {0,1}']

# files = list(file_to_annotations.keys())


name_to_file = {
    'unfun': {
        'RoBERTA-swap': 'unfun_inferences_200/unfun_roberta-base/2024-02-01-21-03-48/model_outputs.tsv',
        'Mistral Instruct': 'unfun_inferences_200/unfun_mistral_instruct/2024-01-30-19-07-49/model_outputs.tsv',
        'ChatGPT-3.5': 'unfun_inferences_200/unfun_chatgpt/2024-01-30-19-07-49/model_outputs.tsv',
        'GPT-4': 'unfun_inferences_200/unfun_gpt4/2024-01-30-19-09-59/model_outputs.tsv',
        'News Headlines': 'real_sample_200/real_heads.tsv',
        'Unfun Players': 'unfun_[GOLD]',
    },
    'satire': {
        'Mistral Instruct': 'satire_inferences_200/satire_mistral_instruct/2024-01-30-19-28-04/model_outputs.tsv',
        'ChatGPT-3.5': 'satire_inferences_200/satire_chatgpt/2024-01-30-19-15-49/model_outputs.tsv',
        'GPT-4': 'satire_inferences_200/satire_gpt4/2024-01-30-19-18-01/model_outputs.tsv',
        'The Onion': 'satire_[GOLD]',
    },
}

UNFUN_ORDER=['RoBERTA-swap', 'Mistral Instruct', 'ChatGPT-3.5', 'GPT-4', 'News Headlines', 'Unfun Players']
SATIRE_ORDER=[ 'Mistral Instruct', 'ChatGPT-3.5', 'GPT-4', 'The Onion']

all_results = []

print('\\midrule')           
print('\\multirow{6}{*}{\\textbf{Unfun}}')
for data_source in UNFUN_ORDER:
    if data_source in ['News Headlines']:
        print('\\cline{2-7}')
    row = get_results(name_to_file['unfun'][data_source], file_to_annotations, key_order)
    all_results.append(['unfun',data_source]+row)

    
    # print(f'& {data_source} & {" & ".join(["%.2f" % round(x,2) if "funniness" in k else "%d" % round(x,0) for k,x in zip(key_order,row)])} \\\\')
    print(f'& {data_source} & ' + " & ".join(["{}".format(round(x*100))+'\%' if "funniness" not in k else "{:.2f}".format(round(x,2)) for k,x in zip(key_order,row)]) +' \\\\')


print('\\midrule')  
print('\\multirow{5}{*}{\\textbf{Humor}}')
for data_source in SATIRE_ORDER:
    if data_source in ['The Onion']:
        print('\\cline{2-7}')
    row = get_results(name_to_file['satire'][data_source], file_to_annotations, key_order)
    all_results.append(['satire',data_source]+row)

   
    # print(f'& {data_source} & {" & ".join(["%.2f" % round(x,2) if "funniness" in k else "$howC0l0AA!!AA!%d" % round(x,0) for k,x in zip(key_order,row)])} \\\\')
    # print(f'& {data_source} & {" & ".join(["%.2f" % round(x,2) for _,x in zip(key_order,row)])} \\\\')

    print(f'& {data_source} & ' + " & ".join(["{}".format(round(x*100))+'\%' if "funniness" not in k else "{:.2f}".format(round(x,2)) for k,x in zip(key_order,row)]) +' \\\\')

print('\\bottomrule')


# save to tsv
df = pd.DataFrame(all_results, columns=['type', 'source']+key_order)
df.to_csv('annotations/annotations_aggregated.tsv', sep='\t', index=False)


    # type_file_labels, file_texts = get_labels(file_to_annotations[file], 'real/satirical/neither {r/s/n}')
    # coherence_labels, _ = get_labels(file_to_annotations[file], 'coherent {0,1}')
    # grammatical_labels, _ = get_labels(file_to_annotations[file], 'grammatical {0,1}')
    # funniness_labels, _ = get_labels(file_to_annotations[file], 'funniness {0,1,2}')

    # is_real, _ = get_labels(file_to_annotations[file], 'is real')
    # is_satirical, _ = get_labels(file_to_annotations[file], 'is satirical')




    # agreement = np.mean([1 if x[0]==x[1] else 0 for x in type_file_labels])
    # all_agreements.append(agreement)
    # print('Agreement:', agreement)

    # unfun_success = []
    # for i in range(len(type_file_labels)):
    #     if 'r' in type_file_labels[i]:
    #         unfun_success.append(1)
    #     elif 's' in type_file_labels[i] and 0 in funniness_labels[i]:
    #         unfun_success.append(1)
    #     elif 'n' in type_file_labels[i] and 1 in coherence_labels[i] and 1 in grammatical_labels[i]:
    #         unfun_success.append(1)
    #     else:
    #         unfun_success.append(0)

    # print('Generous Unfun success:', np.mean(unfun_success)) 



    # print(type_file_labels)
    # at_least_one_r = [x for x in type_file_labels if 'r' in x]
    # at_least_one_s = [x for x in type_file_labels if 's' in x]

    # print('R percentage:', len(at_least_one_r)/len(type_file_labels))
    # print('S percentage:', len(at_least_one_s)/len(type_file_labels))

    # funniness = np.mean([np.mean(x) if x else 0 for x in funniness_labels ])
    # print('Funniness:', funniness)



    # print(coherence_labels)
    # print(grammatical_labels)

    # import pdb; pdb.set_trace()
        
    

# print('Overall agreement:', np.mean(all_agreements))




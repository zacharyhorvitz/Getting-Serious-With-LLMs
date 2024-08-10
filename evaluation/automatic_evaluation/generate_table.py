import json
import numpy as np
import pandas as pd

from edit_distance import find_edit_distance
from compute_ld import compute_lexical_diversity

UNFUN_ORDER=['RoBERTA-swap', 'Mistral', 'Mistral Instruct', 'ChatGPT-3.5', 'GPT-4', 'News Headlines', 'Unfun Players']
SATIRE_ORDER=['Mistral', 'Mistral Instruct', 'ChatGPT-3.5', 'GPT-4', 'The Onion']
CLASSIFIERS = ['mistralaiMistral-7B-v0.1',  'roberta-base'] # 'roberta-large', 'roberta-base']
SCORES_PATH = 'median_scores.tsv'


def normalize(texts):
    texts = [' '.join(x.lower().split()) for x in texts]
    # replace common unicode characters
    texts = [x.replace('’', "'").replace('‘',"'").replace('“', '"').replace('”', '"').replace('—', '-').replace('…', '...').replace('</s>','') for x in texts]
    
    return texts



unfun_pairs = {
    'Unfun Players': '../classification/data/unfun/paired/human_unique_training.tsv',
    'News Headlines': '../classification/data/unfun/unpaired/human_satire_real_news.tsv',
    'RoBERTA-swap': '../unfunning/inferences/unfun_roberta-base/2024-01-30-01-21-11/model_outputs.tsv',
    'GPT-4': '../unfunning/inferences/unfun_gpt4/2024-01-25-23-37-40/model_outputs.tsv',
    'ChatGPT-3.5': '../unfunning/inferences/unfun_chatgpt/2024-01-25-22-52-54/model_outputs.tsv',
    'Mistral Instruct': '../unfunning/inferences/unfun_mistral_instruct/2024-01-27-13-28-52/model_outputs.tsv',
    'Mistral': '../unfunning/inferences/unfun_mistral_completion/2024-01-26-01-08-50/model_outputs.tsv',
}

satire_pair  = {
    'Mistral': '../unfunning/inferences/satire_mistral_completion/2024-01-26-13-00-06/model_outputs.tsv',
    'Mistral Instruct': '../unfunning/inferences/satire_mistral_instruct/2024-01-26-11-14-29/model_outputs.tsv',
    'ChatGPT-3.5': '../unfunning/inferences/satire_chatgpt/2024-01-26-11-15-14/model_outputs.tsv',
    'GPT-4': '../unfunning/inferences/satire_gpt4/2024-01-26-12-12-16/model_outputs.tsv',
    'The Onion': '../classification/data/unfun/paired/human_unique_training.tsv',
}

def load_real_news_samples(file):
    with open(file, "r") as file:
        data = [l.split('\t') for l in file.readlines()]
    return [normalize(['',head]) for _, head, label in data if int(label) == 0]

def load_inferred_samples(file):
    with open(file, "r") as file:
        data = [l.split('\t') for l in file.readlines()]
   
    return [normalize([input,output]) for _,input, output in data]

def load_unfun_samples(file, direction='unfun'):
    with open(file, "r") as file:
        data = [l.split('\t') for l in file.readlines()]
    return [normalize([satire,unfun]) if direction == 'unfun' else normalize([unfun,satire]) for _, unfun, _, satire,_ in data]

def get_unfun_metrics(models, file_mapping):
    model_metrics = {}
    for model in models:
        model_metrics[model] = {}
       

        # else:
        file = file_mapping[model]
        if model == 'Unfun Players':
            model_data = load_unfun_samples(file, direction='unfun')
        elif model == 'The Onion':
            model_data = load_unfun_samples(file, direction='satire')
        elif model == 'News Headlines':
            model_data = load_real_news_samples(file)
        else:
            model_data = load_inferred_samples(file)

        print(file)
        print(model_data[:2])

        if model in ['News Headlines']:
            model_metrics[model]['edit_distance'] = -1
        else:
            edit_distances = [find_edit_distance(*entry) for entry in model_data]
            model_metrics[model]['edit_distance'] = round(np.mean(edit_distances), 2)

        lexical_diversities = compute_lexical_diversity([output for _, output in model_data])
        model_metrics[model]['lexical_diversity'] = round(lexical_diversities['ttr'], 3)



    return model_metrics



results = {'unfun': get_unfun_metrics(UNFUN_ORDER, unfun_pairs), 'satire': get_unfun_metrics(SATIRE_ORDER, satire_pair)}

classifier_scores = pd.read_csv(SCORES_PATH, sep='\t')

for direction, data_source in list(zip(['unfun']*len(UNFUN_ORDER), UNFUN_ORDER)) + list(zip(['satire']*len(SATIRE_ORDER), SATIRE_ORDER)):
    for classifier in CLASSIFIERS:
        data = classifier_scores[(classifier_scores['Model'] == classifier) & (classifier_scores['Training Data'] == f'{direction}/{data_source}')]
        if len(data) == 1:
            row = data.iloc[0]
            results[direction][data_source][classifier] = {'f1': row['F1'], 'accuracy': row['Accuracy'], 'f1_mean':row['F1 mean'], 'acc_mean': row['Accuracy mean'], 'F1 std err':row['F1 std err'], 'Accuracy std err': row['Accuracy std err'], 'F1 std': row['F1 std'], 'Accuracy std': row['Accuracy std']}
        else:
            results[direction][data_source][classifier] = {'f1': '-', 'accuracy': '-'}


GOLD_ACCURACIES = {classifier:results['unfun']['Unfun Players'][classifier]['accuracy'] for classifier in CLASSIFIERS}



print('\\midrule')           
print('\\multirow{7}{*}{\\textbf{Unfun}}')
for data_source in UNFUN_ORDER:
    if data_source in ['News Headlines']:
        print('\\cline{2-6}')
    row = [results['unfun'][data_source]['lexical_diversity'], round(results['unfun'][data_source]['edit_distance'],1)]
    for classifier in CLASSIFIERS:
        # row += ['{}, {}'.format(results['unfun'][data_source][classifier]['f1'], results['unfun'][data_source][classifier]['accuracy'])]
        accuracy = results['unfun'][data_source][classifier]['accuracy']
        str_accuracy = str(accuracy)
        if accuracy != '-':
            accuracy *= 100

            str_accuracy = '{:.1f}'.format(round(accuracy,1))   
            # diff = accuracy - GOLD_ACCURACIES[classifier]*100
            # str_diff = '{:.1f}'.format(round(diff,1))
            std_err = results['unfun'][data_source][classifier]['Accuracy std err']
            str_std_err = '{:.1f}'.format(round(std_err*100,1))
        # row += ['{} ({})'.format(str_accuracy, str_diff) if diff != 0 and accuracy != '-' else str_accuracy]
        row += ['{} ({})'.format(str_accuracy, str_std_err) if accuracy != '-' else str_accuracy]



    print(f'& {data_source} & {" & ".join([str(x) for x in row])} \\\\')

print('\\midrule')  
print('\\multirow{6}{*}{\\textbf{Humor}}')
for data_source in SATIRE_ORDER:
    if data_source in ['The Onion']:
        print('\\cline{2-6}')
    row = [results['satire'][data_source]['lexical_diversity'], round(results['satire'][data_source]['edit_distance'],1)]
    for classifier in CLASSIFIERS:
        # row += ['{}, {}'.format(results['satire'][data_source][classifier]['f1'], results['satire'][data_source][classifier]['accuracy'])]
        # row += [str(results['satire'][data_source][classifier]['accuracy'])]
        accuracy = results['satire'][data_source][classifier]['accuracy']
        str_accuracy = str(accuracy)
        if accuracy != '-':
            accuracy *= 100

            str_accuracy = '{:.1f}'.format(round(accuracy,1))   
            # diff = accuracy - GOLD_ACCURACIES[classifier]*100
            # str_diff = '{:.1f}'.format(round(diff,1))
            std_err = results['satire'][data_source][classifier]['Accuracy std err']
            str_std_err = '{:.1f}'.format(round(std_err*100,1))

        # row += ['{} ({})'.format(str_accuracy, str_diff) if diff != 0 and accuracy != '-' else str_accuracy]
        row += ['{} ({})'.format(str_accuracy, str_std_err) if accuracy != '-' else str_accuracy]



    print(f'& {data_source} & {" & ".join([str(x) for x in row])} \\\\')

print('\\bottomrule')


print()
print()

seeds = 5
# print mean and std for every classifier
for data_source in UNFUN_ORDER:
    print(data_source)
    for classifier in CLASSIFIERS:
        accuracy_mean = results['unfun'][data_source][classifier]['acc_mean']
        std = results['unfun'][data_source][classifier]['Accuracy std']
        print(f'{classifier}: {accuracy_mean} ({std}) ({seeds})')
    
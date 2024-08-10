import glob
import os
import json
import numpy as np
import pandas as pd

# path_to_files = '/burg/nlp/users/zfh2000/UNFUN_EVALS_TEST/*'
path_to_files = '/burg/nlp/users/zfh2000/UNFUN_FINAL/*'
num_seeds = 5


# UNFUN_ORDER=['RoBERTA-swap', 'Mistral', 'Mistral Instruct', 'ChatGPT-3.5', 'GPT-4', 'News Headlines', 'Unfun Players']
# SATIRE_ORDER=['Mistral', 'Mistral Instruct', 'ChatGPT-3.5', 'GPT-4', 'The Onion']


def get_data_model_from_path(path):

    if 'hindi' in path:
        pass

    else:
        if '_mistral_completion_' in path:
            data_model = 'Mistral'
        elif '_mistral_instruct_' in path:
            data_model =  'Mistral Instruct'
        elif '_chatgpt_' in path:
            data_model = 'ChatGPT-3.5'
        elif '_gpt4_' in path:
            data_model = 'GPT-4'
        elif '_satire_real_' in path:
            data_model = 'News Headlines'
        elif '_human_' in path:
            data_model =  'Unfun Players'
        elif '_roberta-base' in path:
            data_model = 'RoBERTA-swap'
        else:
            raise ValueError('Data model not recognized: ' + path)
    
    # if '_fixed' in path:
    #     data_model += ' (fixed)'
    
    return data_model


test_folder_names = None

model_to_metrics = {}
data_configs = set()

for path in glob.glob(path_to_files):
    if 'slurm_jobs' in path:
        continue

    model_name = os.path.basename(path)
    print(model_name)
    if model_name == 'mistralaiMistral-7B-v0.1':
        path = os.path.join(path, 'mistralai')

    model_to_metrics[model_name] = {}

    for experiment in os.listdir(path):
        print(experiment)
        if '_human_' in experiment or '_unfun_' in experiment:
            label = 'unfun'
        elif '_satire_' in experiment:
            label = 'satire'
        # elif '_unfun_' in experiment:
            # label = 'unfun'
        else:
            raise ValueError('Experiment name not recognized.: '+ experiment)
              
        data_model_name = get_data_model_from_path(experiment)
        experiment_name = f'{label}/{data_model_name}'
        data_configs.add(experiment_name)
        
        exp_path = os.path.join(path, experiment)
        run = os.listdir(exp_path)
        assert len(run) == num_seeds, 'More than one run found in experiment folder: ' + exp_path

        if test_folder_names is None:
            test_folder_names = sorted(set([os.path.basename(os.path.dirname(x)) for x in glob.glob(exp_path + '/*/*/*/*test_results.tsv')]))
            print(test_folder_names)
        
        else:
            cur_exp_paths = list(glob.glob(exp_path + '/*/*/*/*test_results.tsv'))
            cur_exp_folders = sorted(set([os.path.basename(os.path.dirname(x)) for x in cur_exp_paths]))

            assert cur_exp_folders == test_folder_names, f'Test folder names do not match: {cur_exp_folders} vs {test_folder_names}'

        model_to_metrics[model_name][experiment_name] = {t_name: {} for t_name in test_folder_names}

        for test_folder in test_folder_names:

            path_query = exp_path + f'/*/*/{test_folder}/*test_results.tsv'
            # print(path_query)
            # exit()           
            result_paths = list(glob.glob(path_query))

            result_paths = [x for x in result_paths] # if '_mistral_completion_' not in x or '2024-01-30' in x]

            # assert len(result_paths) == num_seeds, f'Found {len(result_paths)} result files for {path_query} instead of {num_seeds}'
            assert len(result_paths) == num_seeds, f'Found {len(result_paths)} result files for {path_query} instead of {num_seeds}'

            metrics = {'f1': [], 'acc': []}

            for result_path in result_paths:
                with open(result_path, 'r') as f:
                    result = json.loads(f.readlines()[1].strip().replace("'", '"'))
                
                    f1 = result['test_eval_f1']
                    acc = result['test_eval_accuracy']
                    metrics['f1'].append(f1)
                    metrics['acc'].append(acc)

            # get median
           
            metrics['f1_std'] = np.std(metrics['f1'])
            metrics['acc_std'] = np.std(metrics['acc'])
            metrics['f1_std_err'] = metrics['f1_std'] / np.sqrt(num_seeds)
            metrics['acc_std_err'] = metrics['acc_std'] / np.sqrt(num_seeds)
            metrics['f1_mean'] = np.mean(metrics['f1'])
            metrics['acc_mean'] = np.mean(metrics['acc'])
            metrics['f1'] = np.median(metrics['f1'])
            metrics['acc'] = np.median(metrics['acc'])

            model_to_metrics[model_name][experiment_name][test_folder] = metrics

print(model_to_metrics)


columns = ['Model', 'Training Data', 'F1', 'Accuracy', 'F1 mean', 'Accuracy mean', 'F1 std', 'Accuracy std', 'F1 std err', 'Accuracy std err']
data = []

data_config = sorted(data_configs)
model_names = sorted(model_to_metrics.keys())

for model_name in model_names:
    for data_model in data_config:
        if data_model in model_to_metrics[model_name]:
            metrics = model_to_metrics[model_name][data_model]
            cur_info = [model_name, data_model]
            test_folder = test_folder_names[0]
            cur_info.append(metrics[test_folder]['f1'])
            cur_info.append(metrics[test_folder]['acc'])

            cur_info.append(metrics[test_folder]['f1_mean'])
            cur_info.append(metrics[test_folder]['acc_mean'])

            cur_info.append(metrics[test_folder]['f1_std'])
            cur_info.append(metrics[test_folder]['acc_std'])

            cur_info.append(metrics[test_folder]['f1_std_err'])
            cur_info.append(metrics[test_folder]['acc_std_err'])
            data.append(cur_info)

df = pd.DataFrame(data, columns=columns)
df.to_csv('median_scores.tsv', sep='\t', index=False)
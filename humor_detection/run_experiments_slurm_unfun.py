# Slurm script to train models on the unfun dataset

import subprocess
import os
import time
import itertools

from datetime import datetime
if __name__ == '__main__':

    EXTA_ARGS = [' --project_name unfun_training_paper_final --eval_metric eval_accuracy ']
    SLURM_TEMPLATE = '''#!/bin/bash
#
# Simple "Hello World" submit script for Slurm.
#
#SBATCH --account=nlp         # Replace ACCOUNT with your group account name
#SBATCH --job-name={}     # The job name.
#SBATCH -c 4                      # The number of cpu cores to use
#SBATCH -t 0-04:00                 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=5gb         # The memory the job will use per cpu core
#SBATCH --gres=gpu:1              # Request 1 GPU "generic resource"
#SBATCH -o {} # The file name for the output.

{}

#Command to execute Python program
{}

#End of script
'''
    CONDA_CMD = 'source /burg/nlp/users/zfh2000/miniconda3/bin/activate'
    SLURM_CMD = 'sbatch {}'
    SEEDS = [1234] #, 2345, 3456, 4567, 5678]

    model_to_config = {
        'roberta-base': {
            'lr': 1.25e-05,
            'bs': 32,
        },
        'mistralai/Mistral-7B-v0.1': {
            'lr': 6.25e-06,
            'bs': 32,
        }
    }


    name_to_path = {
        'human': '../datasets/unfun/unfun_processed/unpaired/human_unique_training.tsv',

        'human_satire_real': '../datasets/unfun/unfun_processed/unpaired/human_satire_real_news.tsv',
        
        'roberta_unfun': '../data_generation/inferences/unfun_roberta-base/2024-01-30-01-21-11/model_outputs.tsv_reformatted',

        'gpt4_unfun': '../data_generation/inferences/unfun_gpt4/2024-01-25-23-37-40/model_outputs.tsv_reformatted',
        'gpt4_satire': '../data_generation/inferences/satire_gpt4/2024-01-26-12-12-16/model_outputs.tsv_reformatted',

        'gpt3.5_unfun': '../data_generation/inferences/unfun_chatgpt/2024-01-25-22-52-54/model_outputs.tsv_reformatted',
        'gpt3.5_satire': '../data_generation/inferences/satire_chatgpt/2024-01-26-11-15-14/model_outputs.tsv_reformatted',

        'mistral_instruct_unfun': '../data_generation/inferences/unfun_mistral_instruct/2024-01-27-13-28-52/model_outputs.tsv_reformatted', 
        'mistral_instruct_satire': '../data_generation/inferences/satire_mistral_instruct/2024-01-26-11-14-29/model_outputs.tsv_reformatted', 

        'mistral_completion_unfun_fixed': '../data_generation/inferences/unfun_mistral_completion/2024-01-26-01-08-50/model_outputs.tsv_reformatted',
        'mistral_completion_satire_fixed':  '../data_generation/inferences/satire_mistral_completion/2024-01-26-13-00-06/model_outputs.tsv_reformatted',
    }

    val_path = '../datasets/unfun/unfun_processed/unpaired/val_unique_pairs_no_leakage.tsv'
    test_path = ['../datasets/unfun/unfun_processed/unpaired/test_unique_pairs_no_leakage.tsv', '../datasets/unfun/unfun_processed/unpaired/TEST_human_satire_real_news.tsv']

    TRAINING_FILES = ['human', 'human_satire_real', 'roberta_unfun', 'gpt4_unfun', 'gpt4_satire', 'gpt3.5_unfun', 'gpt3.5_satire', 'mistral_instruct_unfun', 'mistral_instruct_satire', 'mistral_completion_unfun_fixed', 'mistral_completion_satire_fixed']
    
    MODELS = [
    # 'mistralai/Mistral-7B-v0.1',
    'roberta-base',
    ]

    OUTDIR = '/burg/nlp/users/zfh2000/UNFUN_OS_TEST'

    all_combinations = list(itertools.product(TRAINING_FILES, EXTA_ARGS, MODELS, SEEDS))

    total_experiments = len(all_combinations)
    print(f'TOTAL EXPERIMENTS: {total_experiments}')
    # print(f'PER GPU: {total_experiments/len(DEVICES)}')

    cmd_str = '''
python train_classify.py  \
--noisy_training_data  {} \
--batch_size {} --learning_rate {} \
--val_data {}  \
--test_data {} \
--out_dir {} \
{} \
--model_name {} \
--seed {}
'''

    command_script_path = os.path.join(OUTDIR, 'slurm_jobs')
    os.makedirs(command_script_path, exist_ok=True)

    task_id = 0
    for training_files, extra_args, model, seed in all_combinations:
        lr = model_to_config[model]['lr']
        bs = model_to_config[model]['bs']

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_identifier =  f'{task_id}_{model}_{"_".join(training_files.split())}_{seed}'.replace('/', '_')
        training_files = [name_to_path[x] for x in training_files.split()]
        task_str = cmd_str.format(' '.join(training_files), bs, lr, val_path, ' '.join(test_path), OUTDIR, extra_args, model, seed)

        task_script_path = os.path.join(command_script_path, f'{task_identifier}.sh')
        task_str = SLURM_TEMPLATE.format(task_identifier, task_script_path+'.out', CONDA_CMD, task_str)

        with open(task_script_path, 'w') as f:
            f.write(task_str)
        # change permissions
        subprocess.run(f'chmod +x {task_script_path}', shell=True, check=True)

        print(task_str)
        print()
        slurm_cmd = SLURM_CMD.format(task_script_path)
        print('RUNNING: ', slurm_cmd)
        print(task_str)
        print()
        subprocess.run(slurm_cmd, shell=True, check=True)
        time.sleep(10)
        task_id +=1

    





    
    



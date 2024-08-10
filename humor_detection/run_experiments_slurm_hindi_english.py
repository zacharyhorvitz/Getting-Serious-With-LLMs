# Slurm script to train models on the english-hindi code-mixed dataset

import subprocess
import os
import time
import itertools

from datetime import datetime
if __name__ == '__main__':

    # DEVICES = [0, 1, 2, 3]
    EXTA_ARGS = [' --project_name hindi_unfun_eval --eval_metric eval_accuracy']
    SLURM_TEMPLATE = '''#!/bin/bash


# Simple "Hello World" submit script for Slurm.
#
#SBATCH --account=nlp         # Replace ACCOUNT with your group account name
#SBATCH --job-name={}     # The job name.
#SBATCH -c 4                      # The number of cpu cores to use
#SBATCH -t 0-0:59                 # Runtime in D-HH:MM
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
    SEEDS = [1234, 2345, 3456, 4567, 5678]
    LRS = [1.5625e-06] #[5e-5, 2.5e-5, 1.25e-5, 6.25e-6, 3.125e-6, 1.5625e-6] #[1.5625e-6] #
    BS = [8] #[256, 128, 64, 32, 16, 8]

    name_to_path = {
        'human': '../datasets/hindi-english/train_filtered.tsv',
        'gpt4_unfun_filtered_fixed': '../data_generation/hindi_unfuns/filter/train_unfuns_both_reformatted.tsv', 
        'gpt4_unfun_filtered_swapped25': '../data_generation/hindi_unfuns/filter/train_unfuns_just_unfun_reformatted_0.25.tsv',
        'gpt4_unfun_filtered_swapped50': '../data_generation/hindi_unfuns/filter/train_unfuns_just_unfun_reformatted_0.5.tsv',
    }

    val_paths = ['../datasets/hindi-english/dev_filtered.tsv'] 
    test_path = ['../datasets/hindi-english/test_filtered.tsv','../hindi_english_human_eval/unfunned_human_non_humorous.tsv']

    TRAINING_FILES = ['gpt4_unfun_filtered_swapped50', 'gpt4_unfun_filtered_swapped25', 'gpt4_unfun_filtered_swapped50', 'gpt4_unfun_filtered_fixed'] #['human gpt4_unfun_filtered_fixed'] #, 'human gpt4_unfun_500', 'human gpt4_unfun_filtered', 'human gpt4_unfun'] #['human'] #['human gpt4_unfun'] #['human'] #, 'human gpt4_unfun'] #, 'gpt4_unfun', 'gpt4_unfun_filtered', 'human gpt4_unfun', 'human gpt4_unfun_filtered']

   
    MODELS = [
        'l3cube-pune/hing-roberta',
    ]

    OUTDIR = '/burg/nlp/users/zfh2000/HINDI_EVAL_TEST'

    all_combinations = list(itertools.product(TRAINING_FILES, EXTA_ARGS, MODELS, SEEDS, LRS, BS))

    total_experiments = len(all_combinations)
    print(f'TOTAL EXPERIMENTS: {total_experiments}')

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
    for training_files, extra_args, model, seed, lr, bs in all_combinations:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_identifier =  f'{task_id}_{model}_{"_".join(training_files.split())}_{seed}'.replace('/', '_')
        training_files = [name_to_path[x] for x in training_files.split()]
        task_str = cmd_str.format(' '.join(training_files), bs, lr, ' '.join(val_paths), ' '.join(test_path), OUTDIR, extra_args, model, seed)

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
<h1 align='center'> :satisfied: &rarr; Getting Serious about Humor with LLMs &rarr; :neutral_face: </h1>

## Training Humor Classifiers

We train humor classifiers using [`train_classify.py`](./train_classify.py).

To run the experiments described in the paper, you can use the following python scripts for launching Slurm jobs:

- Unfun corpus experiments: [`run_experiments_slurm_unfun.py`](./run_experiments_slurm_unfun.py)
- English-Hindi experiments: [`run_experiments_slurm_hindi_english.py`](./run_experiments_slurm_hindi_english.py)

Individual configurations can be run (without Slurm), see [`example_train.sh`](./example_train.sh).

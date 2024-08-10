
#!/bin/bash
set -x

export HF_DATASETS_CACHE='/burg/nlp/users/zfh2000/hf_cache' #"/mnt/swordfish-pool2/horvitz/hf_cache/"
export HF_HOME='/burg/nlp/users/zfh2000/hf_cache' #'/burg/home/zfh2000/.cache/huggingface'  # #"/mnt/swordfish-pool2/horvitz/hf_cache/"

# export HF_DATASETS_CACHE="/mnt/swordfish-pool2/horvitz/hf_cache/"
# export HF_HOME="/mnt/swordfish-pool2/horvitz/hf_cache/"

#INPUT_DATA_SATIRE=../classification/data/unfun/unpaired/human_unique_training.tsv
# OUT_DIR=inferences/
INPUT_DATA_UNFUN=../human_eval/sample_200/unpaired/pairs_for_unfun_gen.tsv
OUT_DIR=../human_eval/inferences_200/

python roberta_unfun_baseline.py \
    --input_data $INPUT_DATA_UNFUN \
    --out_folder ${OUT_DIR}/unfun_roberta-base --task_type make-unfunny \
    --model_name roberta-base \
    --num_swaps 3

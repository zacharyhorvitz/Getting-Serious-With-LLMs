
#!/bin/bash
set -ex

# export HF_DATASETS_CACHE='/burg/nlp/users/zfh2000/hf_cache' #"/mnt/swordfish-pool2/horvitz/hf_cache/"
# export HF_HOME='/burg/nlp/users/zfh2000/hf_cache' #'/burg/home/zfh2000/.cache/huggingface'  # #"/mnt/swordfish-pool2/horvitz/hf_cache/"

export HF_DATASETS_CACHE="/mnt/swordfish-pool2/horvitz/hf_cache/"
export HF_HOME="/mnt/swordfish-pool2/horvitz/hf_cache/"

# INPUT_DATA_UNFUN=../classification/data/unfun/unpaired/human_unique_training.tsv
# INPUT_DATA_SATIRE=${INPUT_DATA_UNFUN}
# OUT_DIR=inferences/

INPUT_DATA_UNFUN=../human_eval/sample_200/unpaired/pairs_for_unfun_gen.tsv
INPUT_DATA_SATIRE=../human_eval/sample_200/unpaired/pairs_for_satire_gen.tsv
OUT_DIR=../human_eval/inferences_200/

# GPT: Generate unfuns

python hit_llm_generation_v2.py \
   --model 'gpt-3.5-turbo' \
   --input_data  $INPUT_DATA_UNFUN  \
   --out_folder ${OUT_DIR}/unfun_chatgpt \
   --context_size 8 \
   --start_at 0 \
   --max_tokens 64 \
   --number all \
   --top_p 0.85 \
   --temperature 1.0 \
   --task_type 'make-unfunny' \
   --prompt_data ../classification/data/unfun/paired/train_for_prompting.tsv \
   --prompt_path prompts/unfun_dataset/few-shot/unfun_chat_final.json

python hit_llm_generation_v2.py \
   --model 'gpt-4' \
   --input_data  $INPUT_DATA_UNFUN  \
   --out_folder ${OUT_DIR}/unfun_gpt4 \
   --context_size 8 \
   --start_at 0 \
   --max_tokens 64 \
   --number all \
   --top_p 0.85 \
   --temperature 1.0 \
   --task_type 'make-unfunny' \
   --prompt_data ../classification/data/unfun/paired/train_for_prompting.tsv \
   --prompt_path prompts/unfun_dataset/few-shot/unfun_chat_final.json

# GPT: Generate satire

python hit_llm_generation_v2.py \
    --model 'gpt-3.5-turbo' \
    --input_data  $INPUT_DATA_SATIRE  \
    --out_folder ${OUT_DIR}/satire_chatgpt \
    --context_size 8 \
    --start_at 0 \
    --max_tokens 64 \
    --number all \
    --top_p 0.85 \
    --temperature 1.0 \
    --task_type 'make-funny' \
    --prompt_data ../classification/data/unfun/paired/train_for_prompting.tsv \
    --prompt_path prompts/unfun_dataset/few-shot/satire_chat_final.json

python hit_llm_generation_v2.py \
    --model 'gpt-4' \
    --input_data  $INPUT_DATA_SATIRE  \
    --out_folder ${OUT_DIR}/satire_gpt4 \
    --context_size 8 \
    --start_at 0 \
    --max_tokens 64 \
    --number all \
    --top_p 0.85 \
    --temperature 1.0 \
    --task_type 'make-funny' \
    --prompt_data ../classification/data/unfun/paired/train_for_prompting.tsv \
    --prompt_path prompts/unfun_dataset/few-shot/satire_chat_final.json

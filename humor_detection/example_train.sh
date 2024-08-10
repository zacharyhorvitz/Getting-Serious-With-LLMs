#!/bin/sh


# Train a model on the UNFUN dataset using human unfuns + satire
python train_classify.py  \
    --noisy_training_data  ../datasets/unfun/unfun_processed/unpaired/human_unique_training.tsv \
    --batch_size 32 \
    --learning_rate 1.25e-05 \
    --val_data ../datasets/unfun/unfun_processed/unpaired/val_unique_pairs_no_leakage.tsv  \
    --test_data ../datasets/unfun/unfun_processed/unpaired/test_unique_pairs_no_leakage.tsv \
    --out_dir resulting_checkpoint_dir  \
    --project_name unfun_project \
    --eval_metric eval_accuracy  \
    --model_name roberta-base \
    --seed 1234
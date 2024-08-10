import argparse
import os
import shutil

import torch
from torch.utils.data import Dataset
import random


from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, BitsAndBytesConfig
import wandb
import evaluate
import numpy as np
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
import json
import os

from datetime import datetime

import string

def normalize(texts, strip_punc=False):
    texts = [' '.join(x.lower().split()) for x in texts]
    # replace common unicode characters, hacky
    texts = [x.replace('’', "'").replace('‘',"'").replace('“', '"').replace('”', '"').replace('—', '-').replace('…', '...').replace('</s>','') for x in texts]

    if strip_punc:
        texts = [x.translate(str.maketrans('', '', string.punctuation)) for x in texts]
        texts = [' '.join(x.split()) for x in texts]
    return texts

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128, do_normalize=True, strip_punc=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.do_normalize = do_normalize
        self.strip_punc = strip_punc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        retval = {}

        text = self.data[idx]['text']

        if self.do_normalize:
            text = normalize([text], strip_punc=self.strip_punc)[0]

        tokenized = self.tokenizer(text, truncation=True, max_length=self.max_len, return_tensors='pt')
        retval['text'] = text
        retval['input_ids'] = tokenized['input_ids']
        retval['attention_mask'] = tokenized['attention_mask']
        retval['labels'] = self.data[idx]['label']

        return retval


def pad_to_length(x, length, pad_token_id=0):
    return torch.cat([x, pad_token_id*torch.ones((1, length - x.shape[-1]), dtype=torch.long)], dim=-1)
        
def collate_fn(batch, tokenizer):

    input_ids = []
    attention_masks = []
    labels = []

    for sample in batch:
        input_ids.append(sample['input_ids'])
        attention_masks.append(sample['attention_mask'])
        labels.append(sample['labels'])

    max_len = max(x.shape[-1] for x in input_ids)

    for i in range(len(input_ids)):
        input_ids[i] = pad_to_length(input_ids[i], max_len, pad_token_id=tokenizer.pad_token_id)
        attention_masks[i] = pad_to_length(attention_masks[i], max_len, pad_token_id=tokenizer.pad_token_id)

    return {
        'input_ids': torch.cat(input_ids,0),
        'attention_mask': torch.cat(attention_masks,0),
        'labels': torch.tensor(labels)
    }

def compute_metrics(eval_preds):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    recall_metric = evaluate.load("recall")
    precision_metric = evaluate.load("precision")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    accuracy_results = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_results = f1_metric.compute(predictions=predictions, references=labels, average='binary')
    recall_results = recall_metric.compute(predictions=predictions, references=labels, average='binary')
    precision_results = precision_metric.compute(predictions=predictions, references=labels, average='binary')

    return {
        'eval_accuracy': accuracy_results['accuracy'],
        'eval_f1': f1_results['f1'],
        'eval_recall': recall_results['recall'],
        'eval_precision': precision_results['precision'],
    }

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(name, param.numel())
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def load_tsv_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            id, text, label = line.strip().split('\t')
            data.append({
                'id': id,
                'text': text,
                'label': int(label)
            })
    
    return data

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_data', help='path to the test data', nargs='+')
    parser.add_argument('--test_data', help='path to the val data', nargs='+')
    parser.add_argument('--aux_data', help='path to the val data', nargs='+')
    parser.add_argument('--model_name', default='roberta-large', type=str, help='name of the model')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--noisy_training_data', required=True,  nargs='+', help='path to the noisy/synthetic training data')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--out_dir', default='unfun_classifiers', type=str, help='output directory')
    parser.add_argument('--project_name', default='unfun', type=str, help='project name')
    parser.add_argument('--eval_metric', required=True, type=str, help='metric to use for early stopping')
    parser.add_argument('--early_stopping_patience', default=30, type=float, help='patience')
    parser.add_argument('--strip_punc', default=False, action='store_true', help='strip punctuation from the text')
    
    args = parser.parse_args()

    
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    args.out_dir =  os.path.join(args.out_dir, args.model_name.replace('/',''))
    training_names = '-'.join(['_'.join(tname.strip('/').split('/')[-4:]) for tname in args.noisy_training_data])
    run_name = '_'.join([args.model_name, str(args.learning_rate), str(args.batch_size), training_names])
    current_date = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    args.out_dir = os.path.join(args.out_dir, run_name, str(args.seed), current_date)

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, 'args.json'), 'w+') as f:
        json.dump(vars(args), f, indent=2)
    
    noisy_training_data = []
    for training_dname in args.noisy_training_data:
        noisy_training_data.extend(load_tsv_data(training_dname))

    val_data = []
    for val_dname in args.val_data:
        val_data.extend(load_tsv_data(val_dname))

    test_data = []

    for test_dname in args.test_data:
        test_data.append(load_tsv_data(test_dname))

    if args.model_name == "mistralai/Mistral-7B-v0.1":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, quantization_config=bnb_config)
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    bias="none",
                    lora_dropout=0.05,  # Conventional
                    task_type=TaskType.SEQ_CLS,
                )
        
        model = get_peft_model(model, config)
        model.config.pad_token_id = model.config.eos_token_id
        print_trainable_parameters(model)

        use_bf16=True,
        optim="paged_adamw_8bit"
    
    else:
        use_bf16=False
        optim="adamw_torch"
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_eos_token=True)

    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(noisy_training_data[:10])

    train_dataset = TextDataset(noisy_training_data, tokenizer, do_normalize=True, strip_punc=args.strip_punc)
    val_dataset = TextDataset(val_data, tokenizer, do_normalize=True, strip_punc=args.strip_punc)
    test_dataset = [TextDataset(t_data, tokenizer, do_normalize=True, strip_punc=args.strip_punc) for t_data in test_data]


    wandb.init(
        project=args.project_name,
        config={
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'model_name': args.model_name,
            'noisy_training_data': args.noisy_training_data,
            'val_data': args.val_data,
            'accumulation_steps': args.accumulation_steps,
            'eval_metric': args.eval_metric,
            'early_stopping_patience': args.early_stopping_patience,
            'strip_punc': args.strip_punc,
            'use_bf16': use_bf16,
        },
        name=run_name

    )
    
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=200,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='logs',
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=args.eval_metric,
        greater_is_better=False if 'loss' in args.eval_metric else True,
        seed=args.seed,
        bf16=use_bf16,
        optim=optim,
        learning_rate=args.learning_rate,
        report_to="wandb",
        lr_scheduler_type='constant',
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda x: collate_fn(x, tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        compute_metrics=compute_metrics,
    )
    model.config.use_cache = False
    trainer.train()

    if test_dataset:
        trainer.save_model()
        for i, (name, t_data)in enumerate(zip(args.test_data, test_dataset)):
            test_results = trainer.predict(t_data)

            test_results_dir = os.path.join(args.out_dir, f'test_results_{i}')
            os.makedirs(test_results_dir, exist_ok=False)

            with open(os.path.join(test_results_dir, 'test_results.tsv'), 'w+') as f:
                f.write(f"{name}\n{test_results.metrics}\n\n")

            with open(os.path.join(test_results_dir, 'test_predictions.npy'), 'wb') as f:
                np.save(f, test_results.predictions)
            
            with open(os.path.join(test_results_dir, 'test_labels.npy'), 'wb') as f:
                np.save(f, test_results.label_ids)

    # delete all checkpoints
            
    for fname in os.listdir(args.out_dir):
        if 'checkpoint-' in fname:
            shutil.rmtree(os.path.join(args.out_dir, fname))


        if 'model.safetensors' in fname:
            os.remove(os.path.join(args.out_dir, fname))


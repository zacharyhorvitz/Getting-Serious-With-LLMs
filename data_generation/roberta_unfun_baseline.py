
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
import os
import sys

import json
from datetime import datetime
import click

from tqdm import tqdm

import torch

sys.path.append('../utils')
from text_utils import load_tsv_data
import random

def mask_prediction(*, model, tokenizer, input_ids, mask_idx, device='cuda'):

    input_ids = input_ids.to(device)

    # clone 
    input_ids = input_ids.clone()

    # mask
    input_ids[0, mask_idx] = tokenizer.mask_token_id

    # get logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs[0]

    # get predictions
    masked_logits = logits[0][mask_idx]
    masked_predictions = torch.argmax(masked_logits, dim=-1)

    # get probabilities
    masked_probs = torch.nn.functional.softmax(masked_logits, dim=-1)

    return masked_predictions, masked_probs


def make_best_replacement(*, text, model, tokenizer, device='cuda'):
    
    input_ids = tokenizer.encode(text, return_tensors='pt')
    input_ids = input_ids.to(device)

    swaps = []

    for i, value in enumerate(input_ids[0]):
        
        predictions, probs = mask_prediction(model=model, tokenizer=tokenizer, input_ids=input_ids, mask_idx=i, device=device)

        orig_prob = probs[value].item()
        new_prob = probs[predictions].item()

        orig_word = tokenizer.decode(value)
        new_word = tokenizer.decode(predictions)
        ratio = new_prob / orig_prob

        swaps.append((ratio, i, value, predictions, orig_word, new_word, orig_prob, new_prob))

    swaps = sorted(swaps, key=lambda x: x[0], reverse=True)
    if swaps[0][0] > 1.0:
        input_ids[0][swaps[0][1]] = swaps[0][3]

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def normalize(texts):
    texts = [x.lower() for x in texts]
    # replace common unicode characters
    texts = [x.replace('’', "'").replace('‘',"'").replace('“', '"').replace('”', '"').replace('—', '-').replace('…', '...') for x in texts]
    
    return texts


@click.command()
@click.option('--model_name', default='roberta-base')
@click.option('--seed', default=1234)
@click.option('--input_data', required=True, multiple=True, help='Input file')
@click.option('--out_folder', required=True)
@click.option('--task_type', required=True, type=click.Choice(['make-unfunny'])) #, 'make-funny']))
@click.option('--num_swaps', default=3)
def main(model_name,seed,input_data,out_folder, task_type, num_swaps):
    

    info = locals()

    random.seed(seed)

    directory = os.path.join(
        out_folder,
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )
  
    os.makedirs(directory, exist_ok=True)

    with open(os.path.join(directory, "info.json"), 'w+') as info_file:
        json.dump(info, info_file, indent=4)

    if not isinstance(input_data, (tuple, list)):
        input_data = [input_data]

    model = RobertaForMaskedLM.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)

    with open(os.path.join(directory, "model_outputs.tsv"),'w+') as out_file:
        for data in input_data:
            samples = list(load_tsv_data(data))

            for (id, text, label) in tqdm(samples):
                    
                    # if task_type == 'make-funny' and int(label) != 0: continue

                    if task_type == 'make-unfunny' and int(label) != 1: continue
                    
                    text = normalize([text])[0]

                    print(text)

                    edited = text
                    for i in range(num_swaps):
                        edited = make_best_replacement(text=edited, model=model, tokenizer=tokenizer, device=device)
                    edited = normalize([edited])[0]

                    print(f'--> {edited}')

                    out_file.write(f'{id}\t{text}\t{edited}\n')

if __name__ == '__main__':
    main()
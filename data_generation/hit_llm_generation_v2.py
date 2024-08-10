''' hit openai models with data to generate (serious of humorous) headlines '''
import os
import json
from datetime import datetime
import click

from tqdm import tqdm
import sys

import torch

sys.path.append('../utils')
from text_utils import load_tsv_data
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

def normalize(texts):
    texts = [x.lower() for x in texts]
    # replace common unicode characters
    texts = [x.replace('’', "'").replace('‘',"'").replace('“', '"').replace('”', '"').replace('—', '-').replace('…', '...') for x in texts]
    
    return texts

def check_equal(a,b):
    a = ' '.join(a.lower().split())
    b = ' '.join(b.lower().split())
    return a == b  

def get_paired_context(*, input, prompt_data, context_size, ordering):

    samples = [[x[k] for k in ordering] for x in prompt_data]
    samples = [(x,y) for x,y in samples if not check_equal(x,input) and not check_equal(y,input)]

    if len(samples) < context_size:
        raise ValueError(f'len(samples) {len(samples)} < context_size {context_size}')

    return random.sample(samples, context_size)
        
def craft_chat_prompt(*, task_prompt, input, prompt_data, context_size, ordering):
    sample_prompt = [{"role": "system", "content":task_prompt}]
    if context_size:
        assert prompt_data
        sample_context = get_paired_context(input=input, prompt_data=prompt_data, context_size=context_size, ordering=ordering)
        for x,y in sample_context:
            sample_prompt.append({"role": "user", "content": x})
            sample_prompt.append({"role": "assistant","content": y})

    sample_prompt.append({"role": "user", "content":input})
    return sample_prompt

def craft_completion_prompt(*, task_prompt, input, prompt_data, context_size, ordering):
    sample_prompt = task_prompt
    if context_size:
        assert prompt_data
        sample_context = get_paired_context(input=input, prompt_data=prompt_data, context_size=context_size, ordering=ordering)
        for x,y in sample_context:
           sample_prompt += '\n' + x + ' -> ' + y

    sample_prompt += '\n' + input + ' ->'
    return sample_prompt


def hit_mistral_chat(
    *,
    prompt,
    max_tokens,
    model,
    temperature=0,
    top_p=0.0,
    device,
    tokenizer,
    stop,
):
    del stop

    if top_p==0.0:
        sample = False
    else:
        sample = True

    # there is no system role in mistral
    if prompt[0]['role'] == 'system':
        text = prompt[0]['content']
        prompt = prompt[1:]
        prompt[0]['content'] = text + " " +prompt[0]['content']

    # print(prompt)

    encodeds = tokenizer.apply_chat_template(prompt, return_tensors="pt")

    model_inputs = encodeds.to(device)

    # print(model_inputs)
    generated_ids = model.generate(model_inputs, max_new_tokens=max_tokens, do_sample=sample, temperature=temperature, top_p=top_p, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.batch_decode(generated_ids)[0]
    response = response.split('[/INST]')[-1].replace('</s>','').strip()

    return response

def hit_mistral_completion(
    *,
    prompt,
    max_tokens,
    model,
    temperature=0,
    top_p=0.0,
    device,
    tokenizer,
    stop,
):

    del stop

    if top_p==0.0:
        sample = False
    else:
        sample = True

    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_tokens, do_sample=sample, temperature=temperature, top_p=top_p, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.batch_decode(generated_ids)[0]
    response = response.replace(prompt,'').replace('<s>','').split('\n')[0].strip()

    return response

CHAT_MODELS = ['gpt-3.5-turbo', 'gpt-4',"mistralai/Mistral-7B-Instruct-v0.1"]
COMPLETION_MODELS = ["mistralai/Mistral-7B-v0.1"]

# TODO: mixtralai/Mixtral-8x7B mixtralai/Mixtral-7B-Instruct-v0.1

@click.command()
@click.option('--top_p', default=0.0)
@click.option('--max_tokens', default=64)
@click.option('--temperature', default=0.0)
@click.option('--model', required=True)
@click.option('--seed', default=1234)
@click.option('--input_data', required=True, multiple=True, help='Input file')
@click.option('--prompt_data', required=False, help='Example data file, if > context size, will be sampled')
@click.option('--out_folder', required=True)
@click.option('--number', required=True, help='all/<number>')
@click.option('--start_at', default=0)
@click.option('--end_at', default=None)
@click.option('--context_size', default=0, required=True)
@click.option('--prompt_path', required=True)
@click.option('--task_type', required=True, type=click.Choice(['make-funny', 'make-unfunny']))
def main(top_p,max_tokens,temperature,model,seed,input_data,prompt_data,out_folder,number,start_at,end_at,context_size, prompt_path, task_type):
    
    if task_type == 'make-funny':
        ordering = ['serious', 'humorous']
    else:
        ordering = ['humorous', 'serious']

    info = locals()

    with open(prompt_path, 'r') as prompt_file:
        prompt_json = json.load(prompt_file)
        prompt = prompt_json['text']

    if prompt_data is not None:
        # 
        prompt_data =  list(load_tsv_data(prompt_data))
        assert len(prompt_data[0]) == 5, "assumes paired format is (unfun-id unfun satire-id satire link) delimited by tabs"
        prompt_data = [normalize([x[1],x[3]]) for x in prompt_data]
        prompt_data = [{"serious":x[0], "humorous":x[1]} for x in prompt_data]
        print('SAMPLE:',prompt_data[:1])

        if len(prompt_data) > int(context_size):
            print(f'len(prompt_data) {len(prompt_data) } > int(context_size) {int(context_size)}, so will be sampling prompted data')
    
    info['prompt'] = prompt_json

    random.seed(seed)

    directory = os.path.join(
        out_folder,
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )

    assert model in CHAT_MODELS + COMPLETION_MODELS, f'unknown model {model}'

    if model in CHAT_MODELS:
        is_chat = True
    else:
        is_chat = False
    tokenizer = None 
    device = None
    if model.lower().startswith('mistral'):
        if is_chat:
            hit_model = hit_mistral_chat
        else:
            hit_model = hit_mistral_completion
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, use_flash_attention_2=False)
        model.to(device)
    else:
        from gpt_utils import hit_openai_chat_wrapper
        hit_model = hit_openai_chat_wrapper
        

    os.makedirs(directory, exist_ok=True)

    with open(os.path.join(directory, "info.json"), 'w+') as info_file:
        json.dump(info, info_file, indent=4)

    total = 0
    idx = 0

    if not isinstance(input_data, (tuple, list)):
        input_data = [input_data]

    with open(os.path.join(directory, "model_outputs.tsv"),'w+') as out_file:
        for data in input_data:
            samples = list(load_tsv_data(data))

            for (id, text, label) in tqdm(samples):
                    
                    if task_type == 'make-funny' and int(label) != 0: continue

                    if task_type == 'make-unfunny' and int(label) != 1: continue
                    
                    text = normalize([text])[0]
                    
                    if idx < start_at:
                        idx += 1
                        continue

                    if end_at is not None and idx >= end_at:
                        break

                    
                    if number != 'all' and total >= int(number):
                        break
                    
                    if is_chat:
                        sample_prompt = craft_chat_prompt(task_prompt=prompt, input=text, prompt_data=prompt_data, context_size=context_size, ordering=ordering)
                    else:
                        sample_prompt = craft_completion_prompt(task_prompt=prompt, input=text, prompt_data=prompt_data, context_size=context_size, ordering=ordering)

                    print(sample_prompt)
                    response = hit_model(
                        prompt=sample_prompt,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        temperature=temperature,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        stop=['".'],
                    )

                    print(f"--> {response}")
                    total += 1
                    idx += 1

                    out_file.write(f'{id}\t{text}\t{response}\n')

if __name__ == '__main__':

    main()


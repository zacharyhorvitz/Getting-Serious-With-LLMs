import sys
import json
import csv
import re

sys.path.append('../utils')
from gpt_utils import *

def clean(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'www\S+', '', tweet)

    # Remove HTML tags
    tweet = re.sub(r'<.*?>', '', tweet)
    
    return tweet

def craft_chat_prompt(*, task_prompt, sample_context, input):
    sample_prompt = [{"role": "system", "content": task_prompt}]
    if len(sample_context) != 0:
        for x,y in sample_context:
            sample_prompt.append({"role": "user", "content": x})
            sample_prompt.append({"role": "assistant","content": y})

    sample_prompt.append({"role": "user", "content":input})
    return sample_prompt

# Collect few-shot examples
with open("../prompts/hindi_dataset/few-shot/hindi_fun_unfuns_prompt.json", "r") as file:
    data = json.load(file)
file.close()

fun_unfuns = []
for entry in data:
    fun_unfuns.append((entry["humor"], entry["non_humor"]))

def generate_unfuns(input_file_path, output_file_path, task_prompt):
    gpt4_fun_unfuns = []
    tsv_file = csv.reader(open(input_file_path), delimiter='\t')
    
    for line in tsv_file:
        # Ignoring unfunny tweets
        if line[2] == "0":
            continue
        if ".twitter." in line[1]:
            continue
        
        # humorous text only
        humor_text = clean(line[1])
        sample_prompt = craft_chat_prompt(task_prompt=task_prompt, sample_context=fun_unfuns, input=humor_text)
        response = hit_openai_chat(
            prompt=sample_prompt,
            max_tokens=256,
            top_p=0.85,
            temperature=1.0,
            model='gpt-4',
            stop=['".'],
        )
        non_humor_text = response.choices[0].message.content
        gpt4_fun_unfuns.append((humor_text, non_humor_text))
    
    print(len(gpt4_fun_unfuns))
    gpt4_fun_unfuns_dict = [{"humor": item[0], "non_humor": item[1]} for item in gpt4_fun_unfuns]
    with open(output_file_path, "w") as file:
        json.dump(gpt4_fun_unfuns_dict, file, indent=4)
    file.close()

# task_prompt = "Tum kaafi madadagaar sahayak ho jo Hindi tweets mein se mazaak hata kar unko gambhir banati ho. Kripya jawaab Hindi ka tweet sudhaar kar do."
task_prompt = "Kya ye diye hue tweet ka humor wala part hata kar use normal bana sakti ho? Aur jitna ho sake utna punctuation use same rakhne ki koshish karna"

generate_unfuns("../classification/data/hindi-english/train.tsv", "train_unfuns.json", task_prompt=task_prompt)
print("Train file done")

generate_unfuns("../classification/data/hindi-english/dev.tsv", "dev_unfuns.json", task_prompt=task_prompt)
print("Dev file done")

generate_unfuns("../classification/data/hindi-english/test.tsv", "test_unfuns.json", task_prompt=task_prompt)
print("Test file done")

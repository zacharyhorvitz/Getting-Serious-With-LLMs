import sys
import json
import csv
from collections import defaultdict

sys.path.append('../utils')
from gpt_utils import *

def craft_chat_prompt(*, task_prompt, sample_context, input):
    sample_prompt = [{"role": "system", "content": task_prompt}]
    if len(sample_context) != 0:
        for x,y in sample_context:
            sample_prompt.append({"role": "user", "content": x})
            sample_prompt.append({"role": "assistant","content": y})

    sample_prompt.append({"role": "user", "content":input})
    return sample_prompt

# Collect few-shot examples
fun_unfuns_examples = []
tsv_file = csv.reader(open("Filtered Unfuns With Classes.tsv"), delimiter='\t')
for line in tsv_file:
    fun_unfuns_examples.append([line[0], line[2]])
# print(fun_unfuns_examples)

def generate_unfuns(input_file_path, output_file_path, task_prompt):
    with open(input_file_path, "r") as file:
        data = json.load(file)
    file.close()
    
    filtered_tweets = []
    class_freq = defaultdict(lambda: "Other")
    class_freq["Yes"] = 0
    class_freq["No"] = 0
    
    for entry in data:
        unfunned_tweet = entry["non_humor"]
        sample_prompt = craft_chat_prompt(task_prompt=task_prompt, sample_context=fun_unfuns_examples, input=unfunned_tweet)
        response = hit_openai_chat(
            prompt=sample_prompt,
            max_tokens=256,
            top_p=0.85,
            temperature=1.0,
            model='gpt-4',
            stop=['".'],
        )
        reply = response.choices[0].message.content
        class_freq[reply] += 1
        if "No" in reply or "no" in reply:
            filtered_tweets.append((entry["humor"], entry["non_humor"]))
    
    print(class_freq)
    filtered_tweets_dict = [{"humor": tweet[0], "non_humor": tweet[1]} for tweet in filtered_tweets]
    with open(output_file_path, "w") as file:
        json.dump(filtered_tweets_dict, file, indent=4)
    file.close()

# task_prompt = "Tum kaafi madadagaar sahayak ho jo Hindi tweets mein se mazaak hata kar unko gambhir banati ho. Kripya jawaab Hindi ka tweet sudhaar kar do."
task_prompt = "You are a pattern-following assistant used to rigorously determine whether a Hindi tweet is intended to be humorous. Given a Hindi tweet, respond only with either of Yes or No. Yes if it is humoruous and No if it is not humorous"
generate_unfuns("train_unfuns.json", "./filter/train_unfuns.json", task_prompt=task_prompt)
print("Train file done")

generate_unfuns("dev_unfuns.json", "./filter/dev_unfuns.json", task_prompt=task_prompt)
print("Dev file done")

generate_unfuns("test_unfuns.json", "./filter/test_unfuns.json", task_prompt=task_prompt)
print("Test file done")

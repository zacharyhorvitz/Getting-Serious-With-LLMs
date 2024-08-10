import string
import numpy as np
import json


# calculates minimum number of edits to transform humor to non_humor
def find_edit_distance(humor, non_humor):
    humor = humor.translate(str.maketrans('', '', string.punctuation)).lower()
    non_humor = non_humor.translate(str.maketrans('', '', string.punctuation)).lower()
    
    humor_tokens = humor.strip().split()
    non_humor_tokens = non_humor.strip().split()
    
    len_humor = len(humor_tokens)
    len_non_humor = len(non_humor_tokens)
    
    minimum_edits = np.zeros((len_humor + 1, len_non_humor + 1), dtype=int)
    
    for i in range(1, len_humor + 1):
        minimum_edits[i][0] = i
    for j in range(1, len_non_humor + 1):
        minimum_edits[0][j] = j
    
    for i in range(1, len_humor + 1):
        for j in range(1, len_non_humor + 1):
            if humor_tokens[i - 1] == non_humor_tokens[j - 1]:
                minimum_edits[i][j] = minimum_edits[i - 1][j - 1]
            else:
                # (substitution, deletion, insertion)
                minimum_edits[i][j] = 1 + min(minimum_edits[i - 1][j - 1], minimum_edits[i - 1][j], minimum_edits[i][j - 1])
    
    return minimum_edits[len_humor][len_non_humor]

# sent1 = "openai language model"
# sent2 = "openai model for language"
# print(edit_distance(sent1, sent2))

if __name__ == "__main__":

    with open("hindi_fun_unfuns_prompt.json", "r") as file:
        data = json.load(file)

    for entry in data:
        entry["edit_distance"] = str(find_edit_distance(entry["humor"], entry["non_humor"]))

    with open("modified_json_file.json", "w") as file:
        json.dump(data, file, indent=4)

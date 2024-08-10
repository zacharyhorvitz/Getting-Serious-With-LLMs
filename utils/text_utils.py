''' text utils '''

def load_tsv_data(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    return lines
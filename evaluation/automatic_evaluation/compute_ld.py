from lexical_diversity import lex_div as ld

def compute_lexical_diversity(texts):

    tokenized = [ld.tokenize(text) for text in texts]

    # combine all tokenized texts into one list
    tokenized = [item for sublist in tokenized for item in sublist]


    ttr = ld.ttr(tokenized)
    mtld = ld.mtld(tokenized)

    return {'ttr': ttr, 'mtld': mtld}


if __name__ == '__main__':
    text = ["This is a test sentence."]
    print(compute_lexical_diversity(text))
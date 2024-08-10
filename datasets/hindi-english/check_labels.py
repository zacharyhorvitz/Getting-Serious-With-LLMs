import pandas as pd

train = '/burg/home/zfh2000/getting-serious-about-humor/classification/data/hindi-english/train.tsv'
dev = '/burg/home/zfh2000/getting-serious-about-humor/classification/data/hindi-english/dev.tsv'
test = '/burg/home/zfh2000/getting-serious-about-humor/classification/data/hindi-english/test.tsv'

train_new = '/burg/home/zfh2000/getting-serious-about-humor/classification/data/hindi-english/train_new.tsv'
dev_new = '/burg/home/zfh2000/getting-serious-about-humor/classification/data/hindi-english/dev_new.tsv'
test_new = '/burg/home/zfh2000/getting-serious-about-humor/classification/data/hindi-english/test_new.tsv'




# load each dataset
train = pd.read_csv(train, sep='\t', header=None)
dev = pd.read_csv(dev, sep='\t', header=None)
test = pd.read_csv(test, sep='\t', header=None)

# combine
original = pd.concat([train, dev, test], axis=0)

print(original.head())

train_new = pd.read_csv(train_new, sep='\t', header=None)
dev_new = pd.read_csv(dev_new, sep='\t', header=None)
test_new = pd.read_csv(test_new, sep='\t', header=None)

new = pd.concat([train_new, dev_new, test_new], axis=0)

print(original.head())
print(new.head())



# check that they are the same length
# assert len(original) == len(new), f'lengths are not the same: {len(original)} vs {len(new)}'



# check that the labels are the sams

id_to_label_orig = {int(row[0]): row[2] for i, row in original.iterrows()}
id_to_label_new = {int(row[0]): row[2] for i, row in new.iterrows()}



missing = 0
for id, label in id_to_label_orig.items():
    if id not in id_to_label_new:
        print(f'id {id} not in new')
        missing += 1
    else:
        assert label == id_to_label_new[id]


print(f'missing: {missing}')

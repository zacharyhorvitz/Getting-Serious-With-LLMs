import pandas as pd

train = '/burg/home/zfh2000/getting-serious-about-humor/classification/data/hindi-english/train.tsv'
dev = '/burg/home/zfh2000/getting-serious-about-humor/classification/data/hindi-english/dev.tsv'
test = '/burg/home/zfh2000/getting-serious-about-humor/classification/data/hindi-english/test.tsv'

# load each dataset
train = pd.read_csv(train, sep='\t')
dev = pd.read_csv(dev, sep='\t')
test = pd.read_csv(test, sep='\t')

# check lengths
print(len(train))
print(len(dev))
print(len(test))

# total
total = len(train) + len(dev) + len(test)
print(total)

# get ratio of positive to negative in each dataset
train_counts = train['1'].value_counts()
dev_counts = dev['1'].value_counts()
test_counts = test['1'].value_counts()

# get ratios
train_ratio = train_counts[1] / train_counts[0]
dev_ratio = dev_counts[1] / dev_counts[0]
test_ratio = test_counts[1] / test_counts[0]

print(train_ratio)
print(dev_ratio)
print(test_ratio)
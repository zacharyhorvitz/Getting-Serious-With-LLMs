import os
import sys
import random

sys.path.append('../../../utils/')

from text_utils import load_tsv_data

# satire_path = 'paired/human_unique_training.tsv'
satire_path = 'paired/test_unique_pairs_no_leakage.tsv'
satire_data = load_tsv_data(satire_path)

news_path = 'real_news/unfun_real_headlines.tsv'
news_data = load_tsv_data(news_path)

to_avoid = [x[1].lower() for x in load_tsv_data('unpaired/human_satire_real_news.tsv') if int(x[2]) == 0]
print(to_avoid)

def sample_no_duplicates(data, sample_size, to_avoid=[]):
    seen = set(to_avoid)
    
    sampled = []
    while len(sampled) < sample_size:
        head_id, head = random.choice(data)

        head = head.lower()

        if head in seen:
            continue

        seen.add(head)
        sampled.append((head_id, head))
        
    return sampled

random.seed(42)

news_sample = sample_no_duplicates(news_data, len(satire_data), to_avoid)

with open('unpaired/TEST_human_satire_real_news.tsv', 'w') as f:
    for (_, _, satire_id, satire, _), (news_id, news) in zip(satire_data, news_sample):
        f.write(f'{news_id}\t{news}\t0\n')
        f.write(f'{satire_id}\t{satire}\t1\n')

import glob
import json

import random

random.seed(42)

for path in list(glob.glob('*_unfuns.json')) + list(glob.glob('filter*/*.json')):


    if 'reformatted' in path:
        continue

    with open(path, 'r') as f:
        data = json.load(f)


    with open(path.replace('.json','_both_reformatted.tsv'), 'w') as f_both:
        with open(path.replace('.json','_just_unfun_reformatted.tsv'), 'w') as f_unfun:
            for i, sample in enumerate(data):
                humor = ' '.join(sample['humor'].split())
                non_humor = ' '.join(sample['non_humor'].split())

                f_both.write(f'{i}\t{non_humor}\t0\n')
                f_both.write(f'{i}\t{humor}\t1\n')

                f_unfun.write(f'{i}\t{non_humor}\t0\n')


    with open(path.replace('.json','_both_reformatted_500.tsv'), 'w') as f_unfun:
        for i, sample in enumerate(data[:500]):
            humor = ' '.join(sample['humor'].split())
            non_humor = ' '.join(sample['non_humor'].split())

            f_unfun.write(f'{i}\t{non_humor}\t0\n')
            f_unfun.write(f'{i}\t{humor}\t1\n')




                
        
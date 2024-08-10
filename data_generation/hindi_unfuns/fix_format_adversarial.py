import glob
import json

for path in glob.glob('adversarial_test.json'):

    # if 'reformatted' in path:
        # continue

    with open(path, 'r') as f:
        data = json.load(f)



    with open(path.replace('.json','_adv_orig_reformatted.tsv'), 'w') as f_orig:
        with open(path.replace('.json','_adv_reformatted.tsv'), 'w') as f_adv:
            for i, sample in enumerate(data):


                orig = ' '.join(sample['original'].split())
                adv = ' '.join(sample['adversarial'].split())

                f_orig.write(f'{i}\t{orig}\t0\n')
                f_adv.write(f'{i}\t{adv}\t0\n')
            
        
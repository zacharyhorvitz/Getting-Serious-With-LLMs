import os
import sys
sys.path.append('../../utils/')

from text_utils import load_tsv_data

os.makedirs('unpaired', exist_ok=True)
folder = '.'
for filename in os.listdir(folder):
    if filename.endswith('.tsv'):
        data = load_tsv_data(os.path.join(folder, filename))
        with open(os.path.join('unpaired', filename), 'w') as f:
            for sample  in data:
                unfun_id, unfun, satire_id, satire = sample[:4]
                f.write(f'{unfun_id}\t{unfun}\t0\n')
                f.write(f'{satire_id}\t{satire}\t1\n')
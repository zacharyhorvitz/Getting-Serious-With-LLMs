import os
import sys
import glob
sys.path.append('../utils/')

from text_utils import load_tsv_data


unfun_paths = 'inferences/unfun_*/*/model_outputs.tsv'
satire_paths = 'inferences/satire_*/*/model_outputs.tsv'

num_lines = None

for label, paths in zip(['unfun', 'satire'], [unfun_paths, satire_paths]):
    for file in glob.glob(paths):
        data = list(load_tsv_data(file))
        print()
        print(label, file)
        print(data[0])
        
        
        with open(file+'_reformatted', 'w') as f:
            idx = 0
            try: 
       
                for sample in data:
                    if len(sample) == 2:
                        sample.append('')
                        print('WARNING: empty output: ', sample)
                    
                    if label == 'unfun':
                        id, satire, unfun = sample
                    else:
                        id, unfun, satire = sample
                    f.write(f'{id}-unfun\t{unfun}\t0\n')
                    f.write(f'{id}-satire\t{satire}\t1\n')
                    idx += 1
              
            except Exception as e:
                print(file, idx)
                print(e)
                raise e
            
            if num_lines is None:
                num_lines = idx
            else:
                assert num_lines == idx, (num_lines, idx)

        

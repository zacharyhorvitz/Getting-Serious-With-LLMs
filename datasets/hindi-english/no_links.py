import glob as glob

# iterate over all tsv files

for path in glob.glob('*.tsv'):
    # read in the tsv file
    with open(path, 'r') as f:
        data = f.readlines()
    
    # iterate over lines in the tsv file and remove the ones with twitter.com links
    new_data = []
    for line in data:
        if 'twitter.com' not in line:
            new_data.append(line)

    # write the new data to a new file
    with open(path.replace('.tsv', '_no_links.tsv'), 'w') as f:
        f.writelines(new_data)
 
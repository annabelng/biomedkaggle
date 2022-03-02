from datasets import load_dataset

basedir = '/Users/annabelng/Personal Items/Personal/MAP/notebooks'

input_dataset = load_dataset('csv', data_files = basedir + '/data/processed_text.csv')
label_dataset = load_dataset('csv', data_files = basedir + '/data/labels.csv')

print(label_dataset)

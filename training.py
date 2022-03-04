from datasets import load_dataset

basedir = '/Users/annabelng/Personal Items/Personal/MAP/notebooks'

input_dataset = load_dataset('csv', data_files = {
    'train': basedir + '/data/train_text.csv',
    'test': basedir + '/data/test_text.csv'
    })

label_dataset = load_dataset('csv', data_files = {
    'train': basedir + '/data/train_label.csv',
    'test': basedir + '/data/test_label.csv'
    })

print(label_dataset)
'''
train_testvalid = text_dataset['train'].train_test_split(test = 0.2)
test_valid = train_testvalid['test'].train_test_split(test = 0.5)
train_testvalid_text = DatasetDict({
    'train_text': train_testvalid['train'],
    'test_text': test_valid['test'],
    'valid_text': test_valid['train']
    })
print(train_testvalid_text)
'''

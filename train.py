from datasets import load_dataset
import os
from datetime import date
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
import torch.nn as nn
import torch

# load datasets

def train():
    # loading logger
    log = logging.getLogger(__name__)

    # loading datasets from preprocessed csv files
    basedir = '/Users/annabelng/Personal Items/Personal/MAP/notebooks'

    input_dataset = load_dataset('text', data_files = {
        'train': basedir + '/data/train_text.csv',
        'test': basedir + '/data/test_text.csv'
        })

    label_dataset = load_dataset('text', data_files = {
        'train': basedir + '/data/train_label.csv',
        'test': basedir + '/data/test_label.csv'
        })

    print(input_dataset)
    print(label_dataset)

    # loading pubmedbert tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # applying tokenizer to each row
    def encode(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    # tokenizing text
    input_dataset = input_dataset.map(encode, batched = True)

    # training model on tokenized and split data
    import torch

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val) for key, val in self.inputs[idx].items() if key != 'text'}
            item['labels'] = torch.tensor(int(self.labels[idx]['text']))
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(input_dataset['train'], label_dataset['train'])
    test_dataset = Dataset(input_dataset['test'], label_dataset['test'])

    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=9)

    training_args = TrainingArguments(
        output_dir="./results",
        logging_dir = './logs',
        logging_steps = 100,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= train_dataset,
    eval_dataset = test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    train()

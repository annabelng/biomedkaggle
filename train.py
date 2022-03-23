from datasets import load_dataset, load_metric
import os
from datetime import date
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
import torch.nn as nn
import torch
import numpy as np


# load datasets

def train():
    # loading logger
    log = logging.getLogger(__name__)

    # loading datasets from preprocessed csv files
    #basedir = '/Users/annabelng/Personal Items/Personal/MAP/notebooks'
    basedir = os.getcwd()
    #datadir = basedir + '/data'

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
    #tokenizer = AutoTokenizer.from_pretrained("blizrys/biobert-v1.1-finetuned-pubmedqa")
    #tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1-mnli")

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
    metric = load_metric("accuracy")

    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=9)
    #model = AutoModelForSequenceClassification.from_pretrained("blizrys/biobert-v1.1-finetuned-pubmedqa", num_labels=9)
    #model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-large-cased-v1.1-mnli")


    import sklearn.metrics as metrics

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)
    
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=9)

    training_args = TrainingArguments(
        output_dir="./results",
        logging_dir = '/home/runs',
        logging_steps = 100,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.1,
        save_total_limit = 5,
        metric_for_best_model='accuracy'
    )

    trainer = Trainer(
        model_init = model_init,
        args=training_args,
        train_dataset= train_dataset,
        eval_dataset = test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics = compute_metrics,
    )

    best_run = trainer.hyperparameter_search(n_trials=3, direction="maximize")
    #trainer.train()

if __name__ == "__main__":
    train()

from datasets import load_dataset
import os
from datetime import date
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
#import torch.nn as nn
import torch

basedir = '/Users/annabelng/Personal Items/Personal/MAP/notebooks'
input_dataset = load_dataset('csv', data_files = {
        'train': basedir + '/data/train_text.csv',
        'test': basedir + '/data/test_text.csv'
        })

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

# applying tokenizer to each row
def encode(examples):
    return tokenizer(examples, truncation=True, padding='max_length', max_length=512, is_split_into_words=True)
raw_inputs = ['hello my name is', 'I have a cold', 'She has a broken arm']
print(input_dataset['train'].features)

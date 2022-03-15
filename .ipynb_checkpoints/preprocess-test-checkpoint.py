import re
import sklearn
import pandas as pd
import numpy as np
import os

basedir = os.getcwd()
datadir = basedir + '/data'

# reading the gene variants
data = pd.read_csv(datadir + '/test_variants.zip')

# reading the training text
data_text = pd.read_csv(datadir + '/test_text.zip', sep = '\|\|', names = ["ID","Text"],skiprows = 1)

def preprocess(total_text, index, column):
    if type(total_text) is not int:
        string = ""
        # Replace multiple spaces with single space
        total_text = re.sub('\s+',' ', total_text)
        # Converting all the chars into lower-case
        total_text = total_text.lower()

        data_text[column][index] = total_text

no_text = []
for index, row in data_text.iterrows():
    if type(row['Text']) is str:
        preprocess(row['Text'], index, 'Text')
    else:
        no_text.append(index)

# merge gene_var and text based on ID
result = pd.merge(data, data_text, on='ID', how = 'left')
result.dropna()

# shuffle and split into train and test dataframes
#train = result.sample(frac=0.8,random_state=200)
#test = result.drop(train.index).sample(frac=1.0)

# grabbing gene class for test and train
gene = result.filter(['Class'], axis=1)-1
#test_gene = test.filter(['Class'], axis=1)-1

# grabbing text for test and train
text = result.filter(['Text'], axis = 1)
#test_text = test.filter(['Text'], axis = 1)

# turning train dataframes into csv
text.to_csv(datadir + '/submit_text.csv', sep=' ', index=False, header=False)
gene.to_csv(datadir + '/submit_label.csv', sep=' ', index=False, header=False)

# turning test dataframes into csv
#test_text.to_csv(datadir + '/test_text.csv', sep=' ', index=False, header=False)
#test_gene.to_csv(datadir + '/test_label.csv', sep=' ', index=False, header=False)

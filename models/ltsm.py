# extract number and types of medication needed for each patient per month / years

# prepare data for lstm
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# convert data for supervised learning (input, output) pairs
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1] # set number of elements 
    df = DataFrame(data)
	cols, names = list(), list()

    # input sequence (demographics & patient info)
    # collects all historic info 
    for i in range():
    
    # output sequence (readmission probability and medication details)
    for i in range():

    # concatenate into pairs

    # drop rows with NA values


# load dataset
if os.path.exists('/data/cleaned_data.csv'):
    dataset = read_csv('cleaned_data.csv', header=0, index_col=0)
else: 
    raise Exception("no data in folder")
values = dataset.values.astype('float32') # ensure all values are float

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning with one hot encoding
# takes one past value, predicts one value (change based on input size)
reframed = series_to_supervised(scaled, 1, 1) 

# drop irrelevant columns
#! replace this
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True) 
print(reframed.head()) 

# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24 
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, t_test.shape)

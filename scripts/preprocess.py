import pandas as pd
from datetime import datetime
import os

# import data
if os.path.exists('data/diabetic_data.csv'):
    data = pd.read_csv('data/diabetic_data.csv')
    print(f"Successfully loaded dataset")
else:
    raise Exception("no data found")

# separate column categories
id_columns =   ['patient_nbr']
medication_columns = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                        'miglitol', 'troglitazone', 'tolazamide', 'examide',
                        'citoglipton', 'insulin', 'glyburide-metformin',
                        'glipizide-metformin', 'glimepiride-pioglitazone',
                        'metformin-rosiglitazone', 'metformin-pioglitazone']
target_column = 'readmitted'
categorical_columns = []
cleaned_data = data.copy()



#### Preprocess data ####
# handle duplicates and empty rows
data.replace("?", pd.NA, inplace=True)
# data.dropna(inplace=True)
data.drop(columns=['payer_code', 'weight'])
print(f"Size after removing empty rows: {data.shape}")

# store data types for each columns
integer = []
categorical = [] # labels

# process each column and standardize data type
for col in data.columns:
    if col in id_columns or col == target_column:
        continue
    elif pd.to_numeric(data[col], errors='coerce').notna().all():
        integer.append(col)
    else:
        categorical.append(col)
    print(f"integer data: {integer}")
    print(f"categorical data: {categorical}")

#### Encode data ####
# 1. medication --> binary
for col in medication_columns:
    # convert to binary
    data[col] = data[col].apply(lambda x: 0 if x == 'No' else 1)

# 2. target --> binary
if target_column in data.columns:
    mapping = {'NO': 0, '<30': 1, '>30': 2}
    data[target_column] = data[target_column].map(mapping)

# 3. process other categorical data
# convert series of strings to dummy codes
if col in categorical:
    if col not in medication_columns or target_column:
        dummies = pd.get_dummies(data[col], prefix=col, drop_first=False) 
        data = pd.concat([data, dummies], axis=1)

#### Summarize data ####
# summarize first 5 rows
print(data.head(5))
print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

# save cleaned data
# data.to_csv('data/cleaned_data.csv')
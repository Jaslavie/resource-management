import pandas as pd
from datetime import datetime
import os
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# import data
if os.path.exists('data/diabetic_data.csv'):
    data = pd.read_csv('data/diabetic_data.csv')
    print(f"Successfully loaded dataset")
else:
    raise Exception("no data found")

# column categories
id_columns =   ['patient_nbr']
medication_columns = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                        'miglitol', 'troglitazone', 'tolazamide', 'examide',
                        'citoglipton', 'insulin', 'glyburide-metformin',
                        'glipizide-metformin', 'glimepiride-pioglitazone',
                        'metformin-rosiglitazone', 'metformin-pioglitazone']
target_column = 'readmitted'
feature_columns = [col for col in data.columns if col not in id_columns + medication_columns + [target_column]]

# handle duplicates and empty rows
data.replace("?", pd.NA, inplace=True)
data.drop_duplicates(inplace=True)
# data.dropna(inplace=True)
data.drop(columns=['payer_code', 'weight', 'medical_specialty'], inplace=True)
print(f"Size after removing empty rows: {data.shape}")

# fill categorical missing data with most freq value
for col in data.select_dtypes(include=['object']).columns:
    if data[col].isna().sum() > 0:
        most_frequent = data[col].mode()[0]
        data[col].fillna(most_frequent, inplace=True)

# fill numeric missing data with median
for col in data.select_dtypes(include=['number']).columns:
    if data[col].isna().sum() > 0:
        median_value = data[col].median()
        data[col].fillna(median_value, inplace=True)

# encode medication and readmission as binary
for col in medication_columns:
    # convert to binary
    data[col] = data[col].apply(lambda x: 0 if x == 'No' else 1)

if target_column in data.columns:
    mapping = {'NO': 0, '<30': 1, '>30': 2}
    data[target_column] = data[target_column].map(mapping)

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

# summarize first 5 rows
print(data.head(5))
print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

#check for outliers
for col in integer:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
    if outliers > 0:
        print(f"{col}: {outliers} outliers ({outliers/len(data)*100:.2f}%)")
        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        print(f"  Capped outliers for {col}")

# feature engineering
if 'age' in data.columns:
    # convert text age ranges to numeric categories
    age_mapping = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    data['age_numeric'] = data['age'].map(age_mapping)
    integer.append('age_numeric')

# save cleaned data
data.to_csv('data/cleaned_diabetic_data.csv', index=False)

selected_features = [
    'time_in_hospital',
    'number_inpatient',
    'number_emergency',
    'number_diagnoses',
    'insulin',
    'diabetesMed',
    'A1Cresult',
    'age_numeric',
    'gender',
    'num_medications',
    'num_lab_procedures',
    'max_glu_serum',
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id'
]

# prep dataset with only selected features
X = data[selected_features]
y = data[target_column]

# feature importance analysis to validate selection
if len(selected_features) > 0:
    # categorical data that needs encoding
    categorical_features = X.select_dtypes(include=['object']).columns

    # one-hot encode on categorical features
    if not categorical_features.empty:
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # fill NaNs with appropriate values
    X = X.fillna(X.median(numeric_only=True))

selected_data = data[selected_features + [target_column]]
selected_data.to_csv('data/selected_features_data.csv', index=False)

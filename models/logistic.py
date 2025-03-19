# logistic regression to predict probability of readmission
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#### Load preprocessed data ####
data = pd.read_csv('data/selected_features_data.csv')

# Split into training and testing data
X = data.drop(columns=['readmitted'])
y = data['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# evaluate
report = classification_report(y_test, y_pred, output_dict=True)

# print results
print(f"{'precision':>15} {'recall':>10} {'f1-score':>10} {'support':>10}")

for label in sorted(report.keys()):
    if label in ['0', '1', '2', 'NO', '<30', '>30']:
        label_display = label
        if label == '0': label_display = 'NO'
        if label == '1': label_display = '<30'
        if label == '2': label_display = '>30'
        
        print(f"{label_display:>15} {report[label]['precision']:>9.2f} {report[label]['recall']:>9.2f} {report[label]['precision']:>9.2f} {report[label]['support']:>10.0f}")

print(f"{'accuracy':>15} {'':<10} {'':<10} {report['accuracy']:>9.2f} {sum(report[l]['support'] for l in ['0', '1', '2']) if '0' in report else sum(report[l]['support'] for l in ['NO', '<30', '>30']):>10.0f}")
print(f"{'macro avg':>15} {report['macro avg']['precision']:>9.2f} {report['macro avg']['recall']:>9.2f} {report['macro avg']['f1-score']:>9.2f} {sum(report[l]['support'] for l in ['0', '1', '2']) if '0' in report else sum(report[l]['support'] for l in ['NO', '<30', '>30']):>10.0f}")
print(f"{'weighted avg':>15} {report['weighted avg']['precision']:>9.2f} {report['weighted avg']['recall']:>9.2f} {report['weighted avg']['f1-score']:>9.2f} {sum(report[l]['support'] for l in ['0', '1', '2']) if '0' in report else sum(report[l]['support'] for l in ['NO', '<30', '>30']):>10.0f}")
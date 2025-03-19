# logistic regression to predict probability of readmission
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

#### Load preprocessed data ####
data = pd.read_csv('data/selected_features_data.csv')

# Split into training and testing data
X = data.drop('readmitted', axis=1)
y = data['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# logistic regression to predict probability of readmission
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#### Load preprocessed data ####
data = pd.read_csv('data/selected_features_data.csv')

# create a readmission target column
def create_readmission_target(row):
    if row['readmitted_<30'] == 1:
        return '<30'
    elif row['readmitted_>30'] == 1:
        return '>30'
    elif row['readmitted_NO'] == 1:
        return 'NO'
    else:
        return None  # Handle any unexpected cases

data['readmitted'] = data.apply(create_readmission_target, axis=1)

# Define X and y for modeling
X = data.drop(columns=['readmitted_<30', 'readmitted_>30',
                      'readmitted_NO', 'readmitted'])
y = data['readmitted']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CHECK: convert remaining categorical features to numerical
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Fill missing values
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train model
model = LogisticRegression(max_iter=2000, solver='lbfgs')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
basic_report = classification_report(y_test, y_pred, output_dict=True)

# Print results
print(f"{'precision':>15} {'recall':>10} {'f1-score':>10} {'support':>10}")

for label in sorted(basic_report.keys()):
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f"{label:>15} {basic_report[label]['precision']:>9.2f} {basic_report[label]['recall']:>9.2f} {basic_report[label]['f1-score']:>9.2f} {basic_report[label]['support']:>10.0f}")

print(f"{'accuracy':>15} {'':<10} {'':<10} {basic_report['accuracy']:>9.2f} {sum(basic_report[label]['support'] for label in basic_report if label not in ['accuracy', 'macro avg', 'weighted avg']):>10.0f}")
print(f"{'macro avg':>15} {basic_report['macro avg']['precision']:>9.2f} {basic_report['macro avg']['recall']:>9.2f} {basic_report['macro avg']['f1-score']:>9.2f} {sum(basic_report[label]['support'] for label in basic_report if label not in ['accuracy', 'macro avg', 'weighted avg']):>10.0f}")
print(f"{'weighted avg':>15} {basic_report['weighted avg']['precision']:>9.2f} {basic_report['weighted avg']['recall']:>9.2f} {basic_report['weighted avg']['f1-score']:>9.2f} {sum(basic_report[label]['support'] for label in basic_report if label not in ['accuracy', 'macro avg', 'weighted avg']):>10.0f}")

# Use RandomizedSearchCV instead of GridSearchCV for faster results
print("\nPerforming hyperparameter tuning (faster method)...")
param_dist = {
    'C': np.logspace(-3, 3, 7),  # More values but will sample only a few
    'penalty': ['l2'],  # Only use L2 for faster convergence
    'solver': ['lbfgs'],  # lbfgs is typically faster than saga
    'class_weight': ['balanced', None],
    'max_iter': [2000]
}

# RandomizedSearchCV is much faster than GridSearchCV
random_search = RandomizedSearchCV(
    LogisticRegression(multi_class='multinomial'),
    param_distributions=param_dist,
    n_iter=10,  # Only try 10 combinations
    cv=3,  # Fewer folds for speed
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)
better_model = random_search.best_estimator_

print(f"\nBest parameters: {random_search.best_params_}")
y_pred = better_model.predict(X_test)
final_report = classification_report(y_test, y_pred, output_dict=True)

# Print improved model results
print("\nImproved Model Performance:")
print(f"{'precision':>15} {'recall':>10} {'f1-score':>10} {'support':>10}")

for label in sorted(final_report.keys()):
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f"{label:>15} {final_report[label]['precision']:>9.2f} {final_report[label]['recall']:>9.2f} {final_report[label]['f1-score']:>9.2f} {final_report[label]['support']:>10.0f}")

print(f"{'accuracy':>15} {'':<10} {'':<10} {final_report['accuracy']:>9.2f} {sum(final_report[label]['support'] for label in final_report if label not in ['accuracy', 'macro avg', 'weighted avg']):>10.0f}")
print(f"{'macro avg':>15} {final_report['macro avg']['precision']:>9.2f} {final_report['macro avg']['recall']:>9.2f} {final_report['macro avg']['f1-score']:>9.2f} {sum(final_report[label]['support'] for label in final_report if label not in ['accuracy', 'macro avg', 'weighted avg']):>10.0f}")
print(f"{'weighted avg':>15} {final_report['weighted avg']['precision']:>9.2f} {final_report['weighted avg']['recall']:>9.2f} {final_report['weighted avg']['f1-score']:>9.2f} {sum(final_report[label]['support'] for label in final_report if label not in ['accuracy', 'macro avg', 'weighted avg']):>10.0f}")

# Calculate improvement
print("\nModel Improvement:")
basic_weighted_f1 = basic_report['weighted avg']['f1-score']
final_weighted_f1 = final_report['weighted avg']['f1-score']
improvement = (final_weighted_f1 - basic_weighted_f1) / basic_weighted_f1 * 100
print(f"F1-score improvement: {improvement:.2f}%")

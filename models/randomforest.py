import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# directory for outputs
os.makedirs('outputs', exist_ok=True)

# load the selected features data
data = pd.read_csv('data/selected_features_data.csv')

def create_readmission_target(row):
    if row['readmitted_<30'] == 1:
        return '<30'
    elif row['readmitted_>30'] == 1:
        return '>30'
    elif row['readmitted_NO'] == 1:
        return 'NO'
    else:
        return None  # Handle any unexpected cases

# Apply the function to create the target column
data['readmitted_target'] = data.apply(create_readmission_target, axis=1)

# Define X and y for modeling
X = data.drop(columns=['readmitted_<30', 'readmitted_>30',
                      'readmitted_NO', 'readmitted_target'])
y = data['readmitted_target']
# split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#feature scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Training set after SMOTE: {X_train_resampled.shape[0]} samples")
print("Class distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts(normalize=True) * 100)

# basic Random Forest model
print("\nTraining basic Random Forest model...")
rf_basic = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_basic.fit(X_train_resampled, y_train_resampled)

# evaluate basic model
print("\nEvaluating basic model...")
y_pred_basic = rf_basic.predict(X_test)
basic_report = classification_report(y_test, y_pred_basic, output_dict=True)
print(classification_report(y_test, y_pred_basic))


# hyperparameter tuning via GridSearchCV
print("\nPerforming hyperparameter tuning...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 12],
    'min_samples_split': [5],
    'min_samples_leaf': [10],
    'class_weight': ['balanced']
    # high and moderate weighting for minority classes
}

# use StratifiedKFold to preserve class distribution per fold
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',  # balanced performance
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_resampled, y_train_resampled)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# train the final model with the best parameters
print("\nTraining final model with best parameters...")
best_rf = grid_search.best_estimator_
best_rf.fit(X_train_resampled, y_train_resampled)

# save model
joblib.dump(best_rf, 'outputs/best_rf_model.pkl')

# evaluate the final model
print("\nEvaluating final model...")
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)

# classification report
print("\nClassification Report:")
final_report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# calculate improvement
print("\nModel Improvement:")
basic_weighted_f1 = basic_report['weighted avg']['f1-score']
final_weighted_f1 = final_report['weighted avg']['f1-score']
improvement = (final_weighted_f1 - basic_weighted_f1) / basic_weighted_f1 * 100
print(f"F1-score improvement: {improvement:.2f}%")

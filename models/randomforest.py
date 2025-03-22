import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Load the selected features data
print("Loading data...")
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

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#feature scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Basic Random Forest model with class weighting instead of SMOTE
print("\nTraining basic Random Forest model...")
rf_basic = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    class_weight='balanced',  # Use class weights to handle imbalance
    n_jobs=-1,
    oob_score=True  # Enable out-of-bag error estimation
)
rf_basic.fit(X_train, y_train)

# Evaluate basic model
print("\nEvaluating basic model...")
y_pred_basic = rf_basic.predict(X_test)
basic_report = classification_report(y_test, y_pred_basic, output_dict=True)
print(classification_report(y_test, y_pred_basic))

# Print OOB error
print(f"Out-of-bag error estimate: {1 - rf_basic.oob_score_:.4f}")

# Plot OOB error
print("\nPlotting out-of-bag error rates...")
plt.figure(figsize=(10, 6))
plt.plot([100], [1 - rf_basic.oob_score_], 'ro', markersize=10)
plt.axhline(y=1 - rf_basic.oob_score_, color='r', linestyle='--', label=f'OOB Error: {1 - rf_basic.oob_score_:.4f}')
plt.xlabel('Number of Trees')
plt.ylabel('Error Rate')
plt.title('Out-of-Bag Error Rate')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('outputs/rf_oob_error.png')
plt.close()

# Plot learning curve to show training vs validation error
print("\nGenerating learning curve...")
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight='balanced'
    ),
    X_train, y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

# Calculate mean and standard deviation
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.grid(True, linestyle='--', alpha=0.6)
plt.plot(train_sizes, 1 - train_mean, 'o-', color='r', label='Training Error')
plt.plot(train_sizes, 1 - val_mean, 'o-', color='g', label='Validation Error')
plt.fill_between(train_sizes, 1 - (train_mean - train_std), 1 - (train_mean + train_std), alpha=0.1, color='r')
plt.fill_between(train_sizes, 1 - (val_mean - val_std), 1 - (val_mean + val_std), alpha=0.1, color='g')
plt.xlabel('Training set size')
plt.ylabel('Error rate')
plt.title('Training and Validation Error vs Training Size')
plt.legend(loc='best')
plt.savefig('outputs/rf_learning_curve.png')
plt.close()

# Test with a smaller hyperparameter grid for faster execution
print("\nPerforming limited hyperparameter tuning...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 12],
    'min_samples_leaf': [10]
}

# Use StratifiedKFold to preserve class distribution per fold
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',  # balanced performance
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Plot cross-validation scores by min_samples_leaf
cv_results = grid_search.cv_results_
params = cv_results['params']
mean_scores = cv_results['mean_test_score']
std_scores = cv_results['std_test_score']

# Extract parameters for plotting
param_labels = [f"n={p['n_estimators']}, d={p['max_depth']}" for p in params]

plt.figure(figsize=(12, 6))
plt.bar(range(len(param_labels)), 1 - mean_scores, yerr=std_scores, 
        align='center', alpha=0.8, color='skyblue', ecolor='black', capsize=5)
plt.xticks(range(len(param_labels)), param_labels, rotation=45)
plt.xlabel('Hyperparameters')
plt.ylabel('Error Rate (1 - F1 Score)')
plt.title('Cross-Validation Error for Different Hyperparameters')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.savefig('outputs/rf_cv_results.png')
plt.close()

# Train the final model with the best parameters
print("\nTraining final model with best parameters...")
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Save model
joblib.dump(best_rf, 'outputs/best_rf_model.pkl')

# Evaluate the final model
print("\nEvaluating final model...")
y_pred = best_rf.predict(X_test)

# Classification report
print("\nClassification Report:")
final_report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y.unique()), 
            yticklabels=sorted(y.unique()))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('outputs/rf_confusion_matrix.png')
plt.close()

# Calculate improvement
print("\nModel Improvement:")
basic_weighted_f1 = basic_report['weighted avg']['f1-score']
final_weighted_f1 = final_report['weighted avg']['f1-score']
improvement = (final_weighted_f1 - basic_weighted_f1) / basic_weighted_f1 * 100
print(f"F1-score improvement: {improvement:.2f}%")

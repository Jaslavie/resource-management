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

X = data.drop(columns=['readmitted'])
y = data['readmitted']

# split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234, stratify=y
)

#feature scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# handle class imbalance with SMOTE
smote = SMOTE(random_state=1234)
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
    random_state=1234,
    class_weight='balanced',
    n_jobs=-1
)
rf_basic.fit(X_train_resampled, y_train_resampled)

# evaluate basic model
print("\nEvaluating basic model...")
y_pred_basic = rf_basic.predict(X_test)
basic_report = classification_report(y_test, y_pred_basic, output_dict=True)
print(classification_report(y_test, y_pred_basic))


# quick test
quick_param_grid = {
    'max_depth': [6, 10, 15],
    'class_weight': ['balanced', {0: 1, 1: 5, 2: 2}]
}

# Keep other parameters fixed
quick_rf = RandomForestClassifier(
    n_estimators=100,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=1234,
    n_jobs=-1
)

# Stage 1: Run the quick grid search
quick_grid_search = GridSearchCV(
    estimator=quick_rf,  # Use the simplified RF model
    param_grid=quick_param_grid,
    cv=3,  # Reduced folds for speed
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search
quick_grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best parameters
print(f"Best parameters from quick search: {quick_grid_search.best_params_}")
print(f"Best score: {quick_grid_search.best_score_:.4f}")

# Train final model with best parameters
best_quick_rf = quick_grid_search.best_estimator_
best_quick_rf.fit(X_train_resampled, y_train_resampled)

# Evaluate the model
y_pred = best_quick_rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature importance
feature_importances = best_quick_rf.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importances)[::-1]

# Print top features
print("\nTop 10 most important features:")
for i in range(min(10, len(sorted_idx))):
    idx = sorted_idx[i]
    print(f"{feature_names[idx]}: {feature_importances[idx]:.4f}")

# # hyperparameter tuning via GridSearchCV
# print("\nPerforming hyperparameter tuning...")
# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [8, 12],
#     'min_samples_split': [10],
#     'min_samples_leaf': [4],
#     'class_weight': ['balanced', {0: 1, 1: 5, 2: 2}]
#     # high and moderate weighting for minority classes
# }

# # use StratifiedKFold to preserve class distribution per fold
# cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1234)

# grid_search = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
#     param_grid=param_grid,
#     cv=cv,
#     scoring='f1_macro',  # balanced performance
#     n_jobs=-1,
#     verbose=1
# )

# grid_search.fit(X_train_resampled, y_train_resampled)

# print(f"\nBest parameters: {grid_search.best_params_}")
# print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# # train the final model with the best parameters
# print("\nTraining final model with best parameters...")
# best_rf = grid_search.best_estimator_
# best_rf.fit(X_train_resampled, y_train_resampled)

# # save model
# joblib.dump(best_rf, 'outputs/best_rf_model.pkl')

# # evaluate the final model
# print("\nEvaluating final model...")
# y_pred = best_rf.predict(X_test)
# y_pred_proba = best_rf.predict_proba(X_test)

# # classification report
# print("\nClassification Report:")
# final_report = classification_report(y_test, y_pred, output_dict=True)
# print(classification_report(y_test, y_pred))

# # calculate improvement
# print("\nModel Improvement:")
# basic_weighted_f1 = basic_report['weighted avg']['f1-score']
# final_weighted_f1 = final_report['weighted avg']['f1-score']
# improvement = (final_weighted_f1 - basic_weighted_f1) / basic_weighted_f1 * 100
# print(f"F1-score improvement: {improvement:.2f}%")

# # plot confusion matrix
# plt.figure(figsize=(10, 8))
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.savefig('outputs/confusion_matrix.png')
# plt.close()

# # plot ROC curves for each class (one-vs-rest)
# plt.figure(figsize=(10, 8))
# classes = [0, 1, 2]
# class_names = ['No Readmission', '<30 Day Readmission', '>30 Day Readmission']
# colors = ['blue', 'red', 'green']

# for i, color, class_name in zip(classes, colors, class_names):
#     # binary labels for current class vs. rest
#     y_test_binary = (y_test == i).astype(int)

#     # probability for the current class
#     y_score = y_pred_proba[:, i]

#     # ROC curve
#     fpr, tpr, _ = roc_curve(y_test_binary, y_score)
#     roc_auc = roc_auc_score(y_test_binary, y_score)

#     # plot
#     plt.plot(fpr, tpr, color=color, lw=2,
#              label=f'{class_name} (AUC = {roc_auc:.2f})')

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves (One-vs-Rest)')
# plt.legend(loc="lower right")
# plt.savefig('outputs/roc_curves.png')
# plt.close()

# # get feature importances
# feature_importances = best_rf.feature_importances_
# feature_names = X.columns
# sorted_idx = np.argsort(feature_importances)[::-1]

# # plot feature importances
# plt.figure(figsize=(12, 8))
# plt.title("Feature Importances")
# plt.bar(range(min(20, X.shape[1])),
#         [feature_importances[i] for i in sorted_idx[:20]],
#         align="center")
# plt.xticks(range(min(20, X.shape[1])),
#           [feature_names[i] for i in sorted_idx[:20]],
#           rotation=90)
# plt.tight_layout()
# plt.savefig('outputs/feature_importances.png')
# plt.close()

# # print top 15 features
# print("\nTop 10 most important features:")
# for i in range(min(10, len(sorted_idx))):
#     idx = sorted_idx[i]
#     print(f"{feature_names[idx]}: {feature_importances[idx]:.4f}")

# print("\nRandom Forest model training and evaluation complete!")
# print("Results and visualizations saved to the 'outputs' directory.")

# # Example of how to load and use the model for predictions
# print("\nExample: How to use the saved model for predictions:")
# print("""
# # Load the saved model
# rf_model = joblib.load('outputs/best_rf_model.pkl')

# # Prepare new data (must have the same features and preprocessing)
# # new_data = pd.read_csv('new_patients.csv')
# # new_data = pd.get_dummies(new_data, ...)  # Apply the same preprocessing

# # Make predictions
# # predictions = rf_model.predict(new_data)
# # prediction_probabilities = rf_model.predict_proba(new_data)
# """)

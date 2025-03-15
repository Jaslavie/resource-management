# classifies demand level (high or low) for each medication. determines which to prioritize

# import libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

seed = np.random.seed(1234)

# load dataset
# if os.path.exists('/data/cleaned_data.csv'):
#     data = pd.read_csv('/data/cleaned_data.csv')
#     print(f"Successfully loaded dataset")
# else:
#     raise Exception("no data found")


# # identify medication columns
# medication_columns = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
#                       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
#                       'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
#                       'miglitol', 'troglitazone', 'tolazamide', 'examide',
#                       'citoglipton', 'insulin', 'glyburide-metformin',
#                       'glipizide-metformin', 'glimepiride-pioglitazone',
#                       'metformin-rosiglitazone', 'metformin-pioglitazone']


def medication_demand(data, medication, feature_columns):
    """
    train and evaluate XGBoost model for classifying high vs low demand for a specific medication
    returns: dict containing model performance metrics
    """

    # define target variable (1 for high demand, 0 for low demand)
    # high demand('Up' or 'Steady'), Low demand('No' or 'Down')
    target = data[medication].apply(lambda x: 1 if x in ['Up', 'Steady'] else 0)

    # prep feature matrix
    X = data[feature_columns].values
    y = target.values

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # feature importance analysis random forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # get top features for model training
    # CHANGE LATER
    top_n_features = 20  # adjust based on feature importance
    selected_indices = indices[:top_n_features]
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    selected_feature_names = [feature_columns[i] for i in selected_indices]

    # train XGBoost model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric=['logloss', 'auc'],
        use_label_encoder=False,
        random_state=seed
    )
    xgb_model.fit(
        X_train_selected, y_train,
        eval_set=[(X_train_selected, y_train), (X_test_selected, y_test)],
        verbose=False
    )

    # predict on test
    y_pred = xgb_model.predict(X_test_selected)
    y_pred_proba = xgb_model.predict_proba(X_test_selected)[:, 1]

    # hyperparameter tuning
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # smaller grid for faster execution
    param_grid_small = {
        'max_depth': [3, 6, 9],
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1]
    }

    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=seed
        ),
        param_grid=param_grid_small,
        cv=3,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train_selected, y_train)

    # train model with best parameters
    best_xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=seed,
        **grid_search.best_params_
    )

    best_xgb_model.fit(
        X_train_selected, y_train,
        eval_set=[(X_test_selected, y_test)],
        verbose=False
    )

    # evaluate best model
    y_pred_best = best_xgb_model.predict(X_test_selected)
    y_pred_proba_best = best_xgb_model.predict_proba(X_test_selected)[:, 1]

    # save model
    model_filename = f'{medication}_xgboost_model.json'
    best_xgb_model.save_model(model_filename)

    # return performance metrics
    metrics = {
        'medication': medication,
        'accuracy': accuracy_score(y_test, y_pred_best),
        'auc': roc_auc_score(y_test, y_pred_proba_best),
        'best_params': grid_search.best_params_,
        'top_features': selected_feature_names[:5]  # Top 5 features
    }

    return metrics, best_xgb_model


# id_columns = ['encounter_id', 'patient_nbr']

# # Define feature columns (all columns except medication columns and ID columns)
# feature_columns = [col for col in data.columns if col not in id_columns + medication_columns + ['readmitted']]
results = []
models = {}

# determine demand for each medication
for med in medication_columns:
    try:
        metrics, model = medication_demand(data, med, feature_columns)
        results.append(metrics)
        models[med] = model
    except Exception as e:
        print(f"Error analyzing {med}: {e}")

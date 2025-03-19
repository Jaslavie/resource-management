import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# load the selected features data
data = pd.read_csv('data/selected_features_data.csv')

X = data.drop(columns=['readmitted'])
y = data['readmitted']
# binary problem (readmission yes/no)
y_binary = (y > 0).astype(int)  # 0=no readmission, 1=any readmission

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234, stratify=y
)

# binary versions of target
y_train_binary = (y_train > 0).astype(int)
y_test_binary = (y_test > 0).astype(int)

# oversample for binary model
smote_binary = SMOTE(random_state=1234)
X_train_binary_resampled, y_train_binary_resampled = smote_binary.fit_resample(
    X_train, y_train_binary
)

# train the binary model (readmit yes/no)
binary_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight='balanced',
    random_state=1234,
    n_jobs=-1
)
binary_rf.fit(X_train_binary_resampled, y_train_binary_resampled)

# evaluate binary model
y_pred_binary = binary_rf.predict(X_test)
print("\nBinary Readmission Model (Yes/No):")
print(classification_report(y_test_binary, y_pred_binary))

# create a dataset of only readmitted patients for timing prediction
readmit_mask_train = y_train > 0  # Only readmitted patients
X_train_readmit = X_train[readmit_mask_train]
y_train_readmit = y_train[readmit_mask_train]

y_train_readmit_timing = (y_train_readmit == 2).astype(int)

# oversample for timing model
smote_timing = SMOTE(sampling_strategy={
        0: 35000,
        1: 25000
    },
    random_state=1234)
X_train_timing_resampled, y_train_timing_resampled = smote_timing.fit_resample(
    X_train_readmit, y_train_readmit_timing
)

# train the timing model
timing_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight='balanced',
    random_state=1234,
    n_jobs=-1
)
timing_rf.fit(X_train_timing_resampled, y_train_timing_resampled)

# evaluate timing model
readmit_mask_test = y_test > 0
X_test_readmit = X_test[readmit_mask_test]
y_test_readmit_timing = (y_test[readmit_mask_test] == 2).astype(int)

y_pred_timing = timing_rf.predict(X_test_readmit)
print("\nReadmission Timing Model (<30 days vs >30 days):")
print(classification_report(y_test_readmit_timing, y_pred_timing))

# combine the models
def predict_combined(X_new):
    readmit_pred = binary_rf.predict(X_new)

    timing_pred = timing_rf.predict(X_new)
    final_pred = readmit_pred.copy()  # Start with binary prediction (0 or 1)
    final_pred[final_pred == 1] = np.where(
        timing_pred[final_pred == 1] == 0,
        1,  # <30 days
        2   # >30 days
    )

    return final_pred

y_pred_combined = predict_combined(X_test)
print("\nCombined Model Performance:")
print(classification_report(y_test, y_pred_combined))

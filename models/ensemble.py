# combine all models with stacking
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb
import joblib
from sklearn.linear_model import LogisticRegression
import keras

load_model = keras.models.load_model

class StackingEnsemble:
    def __init__(self, medication_name):
        self.medication_name = medication_name
        self.base_models = []
        self.meta_model = None
        self.is_fitted = False
        self.feature_selector = None
        self.scaler = StandardScaler()

    def add_lstm_model(self, model_path):
        #add LSTM model


    def add_xgboost_model(self, model_path):
        #add XGBoost model

    def add_svm_model(self, model_path):
        #add SVM model


    def fit(self, X, y):


    def predict(self, X):


    def predict_probability(self, X):


def train_ensembles(data, medication_columns, feature_columns, base_models_dir='models/'):



# data = pd.read_csv('/data/cleaned_data.csv')
# medication_columns = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
#                         'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
#                         'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
#                         'miglitol', 'troglitazone', 'tolazamide', 'examide',
#                         'citoglipton', 'insulin', 'glyburide-metformin',
#                         'glipizide-metformin', 'glimepiride-pioglitazone',
#                         'metformin-rosiglitazone', 'metformin-pioglitazone']

# id_columns = ['encounter_id', 'patient_nbr']
# feature_columns = [col for col in data.columns if col not in id_columns + medication_columns + ['readmitted']]

ensemble_models = train_ensembles(data, medication_columns, feature_columns)

import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#### Import data ####
if os.path.exists('data/diabetic_data.csv'):
    data = pd.read_csv('data/diabetic_data.csv')
    print(f"Successfully loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")
else:
    raise Exception("no data found")

# Create a copy for cleaned data
cleaned_data = data.copy()

# Define column categories
id_columns = ['patient_nbr', 'encounter_id']
medication_columns = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                     'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                     'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                     'miglitol', 'troglitazone', 'tolazamide', 'examide',
                     'citoglipton', 'insulin', 'glyburide-metformin',
                     'glipizide-metformin', 'glimepiride-pioglitazone',
                     'metformin-rosiglitazone', 'metformin-pioglitazone']
target_column = 'readmitted'

#### Basic Preprocessing ####
# Handle duplicates and missing values
cleaned_data.replace("?", np.nan, inplace=True)
cleaned_data.drop_duplicates(inplace=True)
cleaned_data.drop(columns=['payer_code', 'medical_specialty'], inplace=True)
print(f"Size after removing duplicates and specified columns: {cleaned_data.shape}")

# Fill missing categorical data with most frequent value
for col in cleaned_data.select_dtypes(include=['object']).columns:
    if cleaned_data[col].isna().sum() > 0:
        most_frequent = cleaned_data[col].mode()[0]
        cleaned_data[col].fillna(most_frequent, inplace=True)

# Fill missing numeric data with median
for col in cleaned_data.select_dtypes(include=['number']).columns:
    if cleaned_data[col].isna().sum() > 0:
        median_value = cleaned_data[col].median()
        cleaned_data[col].fillna(median_value, inplace=True)

# Identify column data types
integer_columns = []
categorical_columns = []
float_columns = []

# Categorize columns by data type
for col in cleaned_data.columns:
    if col in id_columns or col == target_column:
        continue
    elif pd.to_numeric(cleaned_data[col], errors='coerce').notna().all():
        if cleaned_data[col].dropna().apply(lambda x: float(x).is_integer()).all():
            integer_columns.append(col)
        else:
            float_columns.append(col)
    else:
        categorical_columns.append(col)

print(f"Integer columns: {integer_columns}")
print(f"Float columns: {float_columns}")
print(f"Categorical columns: {categorical_columns}")

#### Feature Engineering ####
# Age conversion
if 'age' in cleaned_data.columns:
    # Convert text age ranges to numeric categories
    age_mapping = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    cleaned_data['age_numeric'] = cleaned_data['age'].map(age_mapping)
    integer_columns.append('age_numeric')

# Diagnosis codes categorization
diabetes_codes = ['250']
circulatory_codes = ['390', '391', '392', '393', '394', '395', '396', '397', '398', '399',
                    '400', '401', '402', '403', '404', '405', '406', '407', '408', '409',
                    '410', '411', '412', '413', '414', '415', '416', '417', '418', '419',
                    '420', '421', '422', '423', '424', '425', '426', '427', '428', '429',
                    '430', '431', '432', '433', '434', '435', '436', '437', '438', '439',
                    '440', '441', '442', '443', '444', '445', '446', '447', '448', '449',
                    '450', '451', '452', '453', '454', '455', '456', '457', '458', '459']
respiratory_codes = ['460', '461', '462', '463', '464', '465', '466', '467', '468', '469',
                    '470', '471', '472', '473', '474', '475', '476', '477', '478', '479',
                    '480', '481', '482', '483', '484', '485', '486', '487', '488', '489',
                    '490', '491', '492', '493', '494', '495', '496', '497', '498', '499',
                    '500', '501', '502', '503', '504', '505', '506', '507', '508', '509',
                    '510', '511', '512', '513', '514', '515', '516', '517', '518', '519']
digestive_codes = ['520', '521', '522', '523', '524', '525', '526', '527', '528', '529',
                  '530', '531', '532', '533', '534', '535', '536', '537', '538', '539',
                  '540', '541', '542', '543', '544', '545', '546', '547', '548', '549',
                  '550', '551', '552', '553', '554', '555', '556', '557', '558', '559',
                  '560', '561', '562', '563', '564', '565', '566', '567', '568', '569',
                  '570', '571', '572', '573', '574', '575', '576', '577', '578', '579']
kidney_codes = ['580', '581', '582', '583', '584', '585', '586', '587', '588', '589',
               '590', '591', '592', '593', '594', '595', '596', '597', '598', '599']
injury_codes = ['800', '801', '802', '803', '804', '805', '806', '807', '808', '809',
               '810', '811', '812', '813', '814', '815', '816', '817', '818', '819',
               '820', '821', '822', '823', '824', '825', '826', '827', '828', '829',
               '830', '831', '832', '833', '834', '835', '836', '837', '838', '839',
               '840', '841', '842', '843', '844', '845', '846', '847', '848', '849',
               '850', '851', '852', '853', '854', '855', '856', '857', '858', '859',
               '860', '861', '862', '863', '864', '865', '866', '867', '868', '869',
               '870', '871', '872', '873', '874', '875', '876', '877', '878', '879',
               '880', '881', '882', '883', '884', '885', '886', '887', '888', '889',
               '890', '891', '892', '893', '894', '895', '896', '897', '898', '899',
               '900', '901', '902', '903', '904', '905', '906', '907', '908', '909',
               '910', '911', '912', '913', '914', '915', '916', '917', '918', '919',
               '920', '921', '922', '923', '924', '925', '926', '927', '928', '929',
               '930', '931', '932', '933', '934', '935', '936', '937', '938', '939',
               '940', '941', '942', '943', '944', '945', '946', '947', '948', '949',
               '950', '951', '952', '953', '954', '955', '956', '957', '958', '959']
infectious_codes = ['001', '002', '003', '004', '005', '006', '007', '008', '009',
                   '010', '011', '012', '013', '014', '015', '016', '017', '018', '019',
                   '020', '021', '022', '023', '024', '025', '026', '027', '028', '029',
                   '030', '031', '032', '033', '034', '035', '036', '037', '038', '039',
                   '040', '041', '042', '043', '044', '045', '046', '047', '048', '049',
                   '050', '051', '052', '053', '054', '055', '056', '057', '058', '059',
                   '060', '061', '062', '063', '064', '065', '066', '067', '068', '069',
                   '070', '071', '072', '073', '074', '075', '076', '077', '078', '079',
                   '080', '081', '082', '083', '084', '085', '086', '087', '088', '089',
                   '090', '091', '092', '093', '094', '095', '096', '097', '098', '099',
                   '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
                   '110', '111', '112', '113', '114', '115', '116', '117', '118', '119',
                   '120', '121', '122', '123', '124', '125', '126', '127', '128', '129',
                   '130', '131', '132', '133', '134', '135', '136', '137', '138', '139']
neoplasm_codes = ['140', '141', '142', '143', '144', '145', '146', '147', '148', '149',
                 '150', '151', '152', '153', '154', '155', '156', '157', '158', '159',
                 '160', '161', '162', '163', '164', '165', '166', '167', '168', '169',
                 '170', '171', '172', '173', '174', '175', '176', '177', '178', '179',
                 '180', '181', '182', '183', '184', '185', '186', '187', '188', '189',
                 '190', '191', '192', '193', '194', '195', '196', '197', '198', '199',
                 '200', '201', '202', '203', '204', '205', '206', '207', '208', '209',
                 '210', '211', '212', '213', '214', '215', '216', '217', '218', '219',
                 '220', '221', '222', '223', '224', '225', '226', '227', '228', '229',
                 '230', '231', '232', '233', '234', '235', '236', '237', '238', '239']

def categorize_diagnosis(code):
    if pd.isna(code) or code == '?' or str(code).strip() == '':
        return 'Unknown'

    code_str = str(code).strip().split('.')[0]  # Get the part before the decimal
    if code_str in diabetes_codes:
        return 'Diabetes'
    elif any(code_str.startswith(prefix) for prefix in circulatory_codes):
        return 'Circulatory'
    elif any(code_str.startswith(prefix) for prefix in respiratory_codes):
        return 'Respiratory'
    elif any(code_str.startswith(prefix) for prefix in digestive_codes):
        return 'Digestive'
    elif any(code_str.startswith(prefix) for prefix in kidney_codes):
        return 'Kidney'
    elif any(code_str.startswith(prefix) for prefix in injury_codes):
        return 'Injury/Poisoning'
    elif any(code_str.startswith(prefix) for prefix in infectious_codes):
        return 'Infectious'
    elif any(code_str.startswith(prefix) for prefix in neoplasm_codes):
        return 'Neoplasm'
    else:
        return 'Other'

# Apply diagnosis categorization
cleaned_data['diagnosis_group'] = cleaned_data['diag_1'].apply(categorize_diagnosis)
categorical_columns.append('diagnosis_group')

# Create clinically relevant groupings
# Admission type groups
emergency_types = [1, 7]  # emergency, trauma center
urgent_types = [2]  # urgent
elective_types = [3]  # elective
other_adm_types = [4, 5, 6, 8]  # newborn, n/a, NULL, not mapped

cleaned_data['admission_type_group'] = cleaned_data['admission_type_id'].apply(
    lambda x: 'Emergency' if x in emergency_types else
              'Urgent' if x in urgent_types else
              'Elective' if x in elective_types else 'Other')
categorical_columns.append('admission_type_group')

# Discharge disposition groups
home_discharge = [1, 6, 8]  # home, home with home health, home with IV
facility_discharge = [2, 3, 4, 5, 10, 15, 22, 23, 24, 27, 28, 29, 30]  # various facilities
expired = [11, 19, 20, 21]  # all death categories
other_discharge = [7, 9, 12, 13, 14, 16, 17, 18, 25, 26]  # other

cleaned_data['discharge_disposition_group'] = cleaned_data['discharge_disposition_id'].apply(
    lambda x: 'Home' if x in home_discharge else
              'Facility' if x in facility_discharge else
              'Expired' if x in expired else 'Other')
categorical_columns.append('discharge_disposition_group')

# Admission source groups
referral_sources = [1, 2, 3]  # physician, clinic, HMO referral
transfer_sources = [4, 5, 6, 10, 18, 22, 25, 26]  # transfers from other facilities
emergency_sources = [7]  # er
other_sources = [8, 9, 11, 12, 13, 14, 15, 17, 19, 20, 21, 23, 24]  # other

cleaned_data['admission_source_group'] = cleaned_data['admission_source_id'].apply(
    lambda x: 'Referral' if x in referral_sources else
              'Transfer' if x in transfer_sources else
              'Emergency' if x in emergency_sources else 'Other')
categorical_columns.append('admission_source_group')

#### Encoding and Data Transformation ####
# 1. Encode medication columns to binary
for col in medication_columns:
    cleaned_data[col] = cleaned_data[col].apply(lambda x: 0 if x == 'No' else 1)

# 2. Encode target column
if target_column in cleaned_data.columns:
    mapping = {'NO': 0, '<30': 1, '>30': 2}
    # Create a new column for the encoded version to preserve original
    cleaned_data[target_column + '_encoded'] = cleaned_data[target_column].map(mapping)

# 3. Process categorical columns with one-hot encoding
categorical_cols_to_encode = [col for col in categorical_columns 
                             if col not in medication_columns 
                             and col != target_column]

for col in categorical_cols_to_encode:
    # One-hot encode categorical columns
    dummies = pd.get_dummies(cleaned_data[col], prefix=col, drop_first=False)
    cleaned_data = pd.concat([cleaned_data, dummies], axis=1)

# Check for outliers in integer columns
# clip the value so its in a normal range
for col in integer_columns:
    if col in cleaned_data.columns:
        Q1 = cleaned_data[col].quantile(0.25)
        Q3 = cleaned_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"Clipping {outliers} outliers in {col}")
            cleaned_data[col] = cleaned_data[col].clip(lower=lower_bound, upper=upper_bound)

# Save the full cleaned dataset
cleaned_data.to_csv('data/cleaned_diabetic_data.csv', index=False)
print(f"Full cleaned dataset saved with {cleaned_data.shape[0]} rows and {cleaned_data.shape[1]} columns")

#### Create Feature-Selected Dataset ####
# Define selected features based on importance
selected_features = [
    'race',
    'admission_type_group',
    'time_in_hospital',
    'number_inpatient',
    'number_emergency',
    'number_diagnoses',
    'diagnosis_group',
    'insulin',
    'diabetesMed',
    'A1Cresult',
    'age_numeric',
    'gender',
    'num_medications',
    'num_lab_procedures',
    'max_glu_serum',
    'discharge_disposition_group',
    'admission_source_group',
    'readmitted'
]

# update selected features to use the grouped categories instead of IDs
selected_features = list(set(selected_features))

# prep dataset with only selected features
X = cleaned_data[selected_features]
y = cleaned_data[target_column]

# Process categorical features with one-hot encoding
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
if categorical_features:
    # One-hot encoding for categorical variables
    selected_data = pd.get_dummies(X, columns=categorical_features)

# Convert boolean columns to integers (0/1)
bool_columns = selected_data.select_dtypes(include=['bool']).columns
if not bool_columns.empty:
    for col in bool_columns:
        selected_data[col] = selected_data[col].astype(int)

# missing values
missing_values = selected_data.isnull().sum()
if missing_values.sum() > 0:
    selected_data = selected_data.fillna(selected_data.median(numeric_only=True))

# save processed dataset
selected_data.to_csv('data/selected_features_data.csv', index=False)

import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
from joblib import dump
import pandas as pd
import logging
import os
import sys
import sklearn
import matplotlib.pyplot as plt
import time


# Set up logging
log_path = '/well/clifton/users/ncu080/UKB_Project/logs/search_xgb_1.log'
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_path, filemode='w')
logger = logging.getLogger()

def log_and_print(message):
    logger.info(message)
    print(message)

# Check XGBoost version and GPU information
# Log and print versions
log_and_print(f'Python version: {sys.version}')
log_and_print(f'scikit-learn version: {sklearn.__version__}')
log_and_print(f'XGBoost version: {xgb.__version__}')
 
# Load the datasets
log_and_print('Loading and optimizing training and test datasets...')
train_data = pd.read_csv('/well/clifton/projects/ukb_v2/derived/prot_final/train_data.csv')

import numpy as np
# Extract necessary features for the training set
# List of columns to exclude, including the newly added ones
exclude_columns = ['Age', 'Sex', 'PRO_Consort_Participant', 'PRO_WellSampleRun', 
                   'PRO_PlateSampleRun', 'PRO_NProteinMeasured', 'Elevated Waist-Hip Ratio', 
                   'Ethnic Group', 'Haemoglobin', 'MCV', 'Platelet Count', 'White Blood Cell Count', 
                   'Hemaaoglobin (Binary)', 'Anemia', 'Back Pain', 'Chest Pain', 'CRP', 'Calcium', 
                   'TP Result', 'Smoking Status', 'Alcohol Status', 'Cholesterol Result', 'C10AA', 
                   'ins_index', 'ID', 'Myeloma', 'time','GLIPR1', 'NPM1', 'PCOLCE']

# Subset train_data by excluding the listed columns
dtrain = train_data.drop(columns=exclude_columns)

# Prepare y_train suitable for Cox loss function
y_train_xgb = pd.Series(np.where(train_data['Myeloma'] == 0, -train_data['time'], train_data['time']), name='y_cox_loss')


# Count of positive and negative values
positive_count = (y_train_xgb > 0).sum()
negative_count = (y_train_xgb < 0).sum()

log_and_print(f"Positive values count: {positive_count}")
log_and_print(f"Negative values count: {negative_count}")


# Log and print shapes of datasets
log_and_print(f'dtrain shape: {dtrain.shape}')
log_and_print(f'y_train_xgb shape: {y_train_xgb.shape}')

seed = 42  # Set the random seed for reproducibility

# Define parameter grid for GPU usage
param_grid = {
    "learning_rate": [0.1],  # Fixed as it consistently performs well
    "n_estimators": [50, 250, 500],  # Include a lower value to prevent overfitting
    "max_depth": [3, 5, 7],  # Slightly lower values to prevent overfitting
    "min_child_weight": [2, 4, 6],  # Include higher values to prevent overfitting
    "gamma": [0, 0.2],  # Exploring slightly higher values
    "subsample": [0.6, 0.7, 0.8],  # Narrowing around 0.6 and 0.8
    "colsample_bytree": [0.6, 0.8],  # Include a slightly higher value
    "reg_lambda": [1, 4, 8],  # Narrowing around the best performing values
    "reg_alpha": [0, 1, 5]  # Exploring more values around the best performing ones
}  

# Initialize XGBoost Regressor
xgb_cox_baseline = xgb.XGBRegressor(
    objective='survival:cox',
    tree_method='hist',  
    device='cuda', 
    random_state=seed,
    eval_metric="cox-nloglik"
)

# Perform GridSearchCV
log_and_print('Starting GridSearchCV...')
grid_search = GridSearchCV(estimator=xgb_cox_baseline, param_grid=param_grid, cv=5, verbose=3, n_jobs=-1, error_score='raise')

# Fit GridSearchCV
grid_search.fit(dtrain, y_train_xgb)
log_and_print('GridSearchCV completed.')

# Get best estimator
best_clf = grid_search.best_estimator_

# Log best parameters
log_and_print(f'Best parameters: {grid_search.best_params_}')

# Save the best model as PKL
model_path_pkl = '/well/clifton/users/ncu080/UKB_Project/models/search_xgb_gs1.pkl'
dump(best_clf, model_path_pkl)
log_and_print(f'Model saved to {model_path_pkl}')

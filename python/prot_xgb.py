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
log_path = '/well/clifton/users/ncu080/UKB_Project/logs/prot_xgb.log'
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
train = pd.read_csv('/well/clifton/projects/ukb_v2/derived/prot_final/train_imp_simple.csv')

import numpy as np
# Extract necessary features for the training set
features = ['SLAMF7', 'TNFRSF17', 'QPCT', 'LY9', 'SLAMF1', 'CNTN5', 'TNFRSF13B',
    'TNFSF13', 'TNFSF13B', 'TIMP1']

dtrain = train[features]

# Prepare y_train suitable for Cox loss function
y_train_xgb = pd.Series(np.where(train['Myeloma'] == 0, -train['time'], train['time']), name='y_cox_loss')


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
    "learning_rate": [0.01],  # Fixed as it consistently performs well
    "n_estimators": [10, 20, 50, 100],  # Include a lower value to prevent overfitting
    "max_depth": [2, 3, 4],  # Slightly lower values to prevent overfitting
    "min_child_weight": [2, 4, 6],  # Include higher values to prevent overfitting
    "subsample": [0.5, 0.6],  # Narrowing around 0.6 and 0.8
    "reg_lambda": [4, 8],  # Narrowing around the best performing values
    "reg_alpha": [0, 1, 10]  # Exploring more values around the best performing ones
} 

# Initialize XGBoost Regressor
xgb_baseline = xgb.XGBRegressor(
    objective='survival:cox',
    tree_method='hist',  
    device='cuda', 
    random_state=seed,
    eval_metric="cox-nloglik"
)

# Perform GridSearchCV
log_and_print('Starting GridSearchCV...')
grid_search = GridSearchCV(estimator=xgb_baseline, param_grid=param_grid, cv=5, verbose=3, n_jobs=-1, error_score='raise')

# Fit GridSearchCV
grid_search.fit(dtrain, y_train_xgb)
log_and_print('GridSearchCV completed.')

# Get best estimator
best_clf = grid_search.best_estimator_

# Log best parameters
log_and_print(f'Best parameters: {grid_search.best_params_}')

# Save the best model as PKL
model_path_pkl = '/well/clifton/users/ncu080/UKB_Project/models/prot_xgb_gs.pkl'
dump(best_clf, model_path_pkl)
log_and_print(f'Model saved to {model_path_pkl}')

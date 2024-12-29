# proteoML
For the paper "Combining High Throughput Plasma Proteomics and Machine Learning for Improved Risk Prediction and Understanding of Multiple Myeloma"

## notebooks

### proteomics_workbook.ipynb
This workbook outlines the pre-porcessing and development of the feature selection XGBoost model

### proteomics_cox_models.ipynb
The workbook outlines the processing (inlcuding multiple imputation) and development of the three Cox models as well as their diagnositcs and performance metrics 

### proteomics_gxb.ipynb
This workbook presents the sensitivity analysis of the 


## Python
Contains the scripts:
### search_gxb.py
First part of the grid search to tune the XGBoost model for feature selection
### search_gxb_2.py
Second part of the grid search to tune the XGBoost model for feature selection
### clin_prot_xgb.py
Grid search to tune the XGBoost model with clinical and proteomic features (n=20)
### prot_xgb.py
Grid search to tune the XGBoost model with clinical and proteomic features (n=10)
### rename_features.py
A simple python script to rename the features

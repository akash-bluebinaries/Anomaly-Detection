import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

df = pd.read_csv('D:/bmsAnomalyDetection/application/Anomaly-detection/notebook/data/labelled_data.csv')

from pycaret.classification import *
from imblearn.over_sampling import RandomOverSampler


ignore = ['Unnamed: 0','Anomaly_Score','timestamps']

# # Initialize the setup
# clf1 = setup(data=df,target='Anomaly',session_id=123,train_size=0.7,preprocess=True,imputation_type='simple',
#              categorical_imputation='mode',numeric_imputation='mean',normalize=True,ignore_features=ignore,
#              normalize_method='robust',combine_rare_levels=True,rare_level_threshold=0.1,data_split_stratify=True,
#              fold_strategy='stratifiedkfold',fold=3,fold_shuffle=True,n_jobs=None,silent=True,fix_imbalance=True,
#              fix_imbalance_method=RandomOverSampler(),ignore_low_variance=True,handle_unknown_categorical=True,
#              unknown_categorical_method='least_frequent',feature_selection=True,feature_selection_method= "classic",
#              log_experiment=True,experiment_name='Pycaret_Experiment001')

import mlflow

# mlflow.set_experiment(experiment_name='training_experiment_1')

with mlflow.start_run(run_name='test002'):
    clf1 = setup(data=df,target='Anomaly',session_id=123,train_size=0.7,
             preprocess=True,
             imputation_type='simple',
             categorical_imputation='mode',
             numeric_imputation='mean',
             normalize=True,
             ignore_features=ignore,
             normalize_method='robust',
             combine_rare_levels=True,
             rare_level_threshold=0.1,
             data_split_stratify=True,
             fold_strategy='stratifiedkfold',
             fold=3,
             fold_shuffle=True,
             n_jobs=None,
             silent=True,fix_imbalance=True,
             fix_imbalance_method=RandomOverSampler(),
             ignore_low_variance=True,
             handle_unknown_categorical=True,
             unknown_categorical_method='least_frequent',
             feature_selection=True,
             feature_selection_method= "classic",
             log_experiment=True,experiment_name='Pycaret_Experiment002'
             )
    best = compare_models(include=['lightgbm','ada','rf'])
    mlflow.end_run()

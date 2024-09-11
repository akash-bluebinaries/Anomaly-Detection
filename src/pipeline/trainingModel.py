import os,sys
import pandas as pd
import numpy as np
from pycaret.classification import *
from imblearn.over_sampling import RandomOverSampler
import mlflow


def training():
    mlflow.set_tracking_uri("http://localhost:5000")

    df = pd.read_csv('D:/bmsAnomalyDetection/application/Anomaly-detection/artifacts/data/labelled_data.csv')
    ignore = ['Unnamed: 0','Anomaly_Score','timestamps']

    # Initialize the setup
    clf1 = setup(data=df,target='Anomaly', session_id=123, train_size=0.7, preprocess=True, imputation_type='simple',
                categorical_imputation='mode', numeric_imputation='mean', normalize=True, ignore_features=ignore,
                normalize_method='robust', combine_rare_levels=True, rare_level_threshold=0.1, data_split_stratify=True,
                fold_strategy='stratifiedkfold',fold=3, fold_shuffle=True,n_jobs=None,silent=True,fix_imbalance=True,
                fix_imbalance_method=RandomOverSampler(),ignore_low_variance=True,handle_unknown_categorical=True,unknown_categorical_method='least_frequent',
                feature_selection=True,feature_selection_method= "classic",log_experiment=True, experiment_name="Model Training",log_plots=True
                )

    # Comparing models to find the best one
    best = compare_models(sort = 'recall')

    # Save the model
    save_model(best, 'D:/bmsAnomalyDetection/application/Anomaly-detection/artifacts/trainedModel')


if __name__== '__main__':
    training()
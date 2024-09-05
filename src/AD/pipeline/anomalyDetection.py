import os,sys
import pandas as pd
import numpy as np
from pycaret.anomaly import *


def detector():

    df = pd.read_csv('D:/bmsAnomalyDetection/application/Anomaly-detection/notebook/data/23-07-24.csv')


    high_cardinality_features = ['timestamps','BMS_state','OBC_port_status','OBC_overvoltage_fault','OBC_overcurrent_fault','OBC_port_weld_fault',
                                'BMS_max_cell_temp_id','BMS_min_cell_temp_id','BMS_max_cell_voltage_id','BMS_min_cell_voltage_id','OBC_mux']


    detect = setup(data=df,
                  session_id=123,
                  preprocess=True,
                  numeric_imputation="median",
                  categorical_imputation="mode",pca=True,
                  pca_components=20,
                  pca_method="linear",
                  normalize=True,
                  high_cardinality_features=high_cardinality_features,
                  high_cardinality_method='frequency',
                  silent=True)


    svm = create_model('svm')
    svm_results = assign_model(svm)

    save_model(svm,'D:/bmsAnomalyDetection/application/Anomaly-detection/artifacts/detectorModel')

    svm_results.to_csv('D:/bmsAnomalyDetection/application/Anomaly-detection/artifacts/data/labelled_data.csv')



if __name__ == '__main__':
    detector()
    



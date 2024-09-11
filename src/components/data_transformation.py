import os,sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_variables = [
                'BMS_state',
                'BMS_max_cell_temp_id',
                'BMS_min_cell_temp_id',
                'BMS_max_cell_voltage_id',
                'BMS_min_cell_voltage_id',
                'OBC_mux',
                'OBC_port_status',
                'OBC_overvoltage_fault',
                'OBC_overcurrent_fault',
                'OBC_port_weld_fault'
                ]

            numerical_variables = ['BMS_soc','BMS_soh', 'BMS_bus_voltage', 'BMS_bus_current', 
                                   'BMS_isolation', 'BMS_max_cell_temp', 'BMS_min_cell_temp',
                                   'BMS_max_cell_voltage', 'BMS_min_cell_voltage', 'LV_soc',
                                   'LV_soh', 'LV_voltage', 'LV_current', 'LV_temperature',
                                   'MCU_motor_speed', 'MCU_motor_avg_temp', 'OBC_output_voltage',
                                   'OBC_output_current','OBC_internal_voltage','OBC_internal_current'
                                   ]
            
            numeric_transformer = Pipeline(steps=[
                 ('imputer', SimpleImputer(strategy='mean')),
                 ('scaler', StandardScaler())
                 ])

            # Categorical transformations: Impute missing and OneHotEncode

            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])

            # Use ColumnTransformer to apply transformations

            preprocessor = ColumnTransformer(
                transformers=[
                ('num', numeric_transformer, numerical_variables),
                ('cat', categorical_transformer, categorical_variables)]
                )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df['BMS_max_cell_temp_id'] = train_df['BMS_max_cell_temp_id'].astype('object')
            train_df['BMS_state'] = train_df['BMS_state'].astype('object')
            train_df['BMS_min_cell_temp_id'] = train_df['BMS_min_cell_temp_id'].astype('object')
            train_df['BMS_max_cell_voltage_id'] = train_df['BMS_max_cell_voltage_id'].astype('object')
            train_df['BMS_min_cell_voltage_id'] = train_df['BMS_min_cell_voltage_id'].astype('object')
            train_df['OBC_mux'] = train_df['OBC_mux'].astype('object')
            train_df['OBC_port_status'] = train_df['OBC_port_status'].astype('object')
            train_df['OBC_overvoltage_fault'] = train_df['OBC_overvoltage_fault'].astype('object')
            train_df['OBC_overcurrent_fault'] = train_df['OBC_overcurrent_fault'].astype('object')
            train_df['OBC_port_weld_fault'] = train_df['OBC_port_weld_fault'].astype('object')

            test_df['BMS_max_cell_temp_id'] = test_df['BMS_max_cell_temp_id'].astype('object')
            test_df['BMS_state'] = test_df['BMS_state'].astype('object')
            test_df['BMS_min_cell_temp_id'] = test_df['BMS_min_cell_temp_id'].astype('object')
            test_df['BMS_max_cell_voltage_id'] = test_df['BMS_max_cell_voltage_id'].astype('object')
            test_df['BMS_min_cell_voltage_id'] = test_df['BMS_min_cell_voltage_id'].astype('object')
            test_df['OBC_mux'] = test_df['OBC_mux'].astype('object')
            test_df['OBC_port_status'] = test_df['OBC_port_status'].astype('object')
            test_df['OBC_overvoltage_fault'] = test_df['OBC_overvoltage_fault'].astype('object')
            test_df['OBC_overcurrent_fault'] = test_df['OBC_overcurrent_fault'].astype('object')
            test_df['OBC_port_weld_fault'] = test_df['OBC_port_weld_fault'].astype('object')
            test_df['Anomaly'] = test_df['Anomaly'].astype('object')
            test_df['Anomaly'] = test_df['Anomaly'].astype('object')

            train_df = train_df.drop(['Anomaly_Score','timestamps'], axis = 1)
            test_df = test_df.drop(['Anomaly_Score','timestamps'], axis = 1)

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "Anomaly"

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]


            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path, obj= preprocessing_obj)

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        


# if __name__ == '__main__':
#     obj = DataTransformation()
#     train_arr,test_arr,_= obj.initiate_data_transformation('artifacts/train.csv','artifacts/test.csv')

import os,sys
import pandas as pd
from pycaret.anomaly import *
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class AnomalyDetectionConfig:
    labelledData_path : str = os.path.join('artifacts','labelledData.csv')
    anomalyDetector_path : str = os.path.join('artifacts','anomalyDetector.pkl')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')


class AnomalyDetection:
    def __init__(self):
        self.detectionConfig = AnomalyDetectionConfig()

    def initiate_anomaly_detection(self):
        logging.info('Initializing Anomaly Detection')
        
        try:
            df = pd.read_csv('D:/bmsAnomalyDetection/application/Anomaly-detection/artifacts/unlabelledData.csv')
            high_cardinality_features = ['timestamps','BMS_state','OBC_port_status','OBC_overvoltage_fault',
                                         'OBC_overcurrent_fault','OBC_port_weld_fault','BMS_max_cell_temp_id',
                                         'BMS_min_cell_temp_id','BMS_max_cell_voltage_id','BMS_min_cell_voltage_id',
                                         'OBC_mux']
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

            svm_results.to_csv(self.detectionConfig.labelledData_path, index=False, header=True)
            save_object(
                file_path=self.detectionConfig.anomalyDetector_path,
                obj=svm)
            
            logging.info('Initiating train test split')
            train_df, test_df = train_test_split(svm_results, test_size=0.3, random_state=101, stratify=svm_results['Anomaly'])
            train_df.to_csv(self.detectionConfig.train_data_path, index=False, header=True)
            test_df.to_csv(self.detectionConfig.test_data_path, index=False, header=True)



            logging.info('Anomaly Detection Successful')
            return (
                self.detectionConfig.labelledData_path,
                self.detectionConfig.anomalyDetector_path,
                self.detectionConfig.train_data_path,
                self.detectionConfig.test_data_path
                )

        except Exception as e:
            raise CustomException(e,sys)
    

        



if __name__ == '__main__':
    obj = AnomalyDetection()
    obj.initiate_anomaly_detection()
    



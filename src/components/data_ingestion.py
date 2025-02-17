import os,sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts','unlabelledData.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Reading data as dataframe')
            df = pd.read_csv('notebook/data/23-07-24.csv')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            logging.info('Saving Raw Data')
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header= True)

            logging.info('Data Ingestion Successful')
            return (
                self.ingestion_config.raw_data_path
                )

        except Exception as e:
            raise CustomException(e,sys)
            

if __name__=='__main__':
    obj= DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()


    logging.info('2. ENTERING DATA_TRANSFORMATION MODULE')
    data_transformation=DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)


    logging.info('3. ENTERING MODEL_TRAINER MODULE')
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer("artifacts/train.csv","artifacts/test.csv"))


            
            
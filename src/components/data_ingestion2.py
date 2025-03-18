#Take data from somewhere and also divide into train - test - validation here itself

import os
import sys

#If sys isn't being recognized
# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
# sys.path.append(root_dir)
#If it ain't recognizing src.components.data_transformation still, then try command -> python -m src.components.data_ingestion2
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation2 import DataTransformation
from src.components.data_transformation2 import DataTransformationConfig

from src.components.model_trainer2 import ModelTrainerConfig
from src.components.model_trainer2 import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(r'D:\Parkinsons\Dataset\PD_vs_DD.csv')
            logging.info('Have read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train test split is initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data is complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    feature_selection_method = "PFI"  # Change this as needed
    train_arr, test_arr, _, selected_features = data_transformation.initiate_data_transformation(train_data, test_data, feature_selection_method=feature_selection_method, k=29)
    print(f"Selected Features: {selected_features}")

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    
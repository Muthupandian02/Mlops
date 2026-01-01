import pandas as pd
import sys
from src.logger import logging
from src.exception import CustomeException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import os

from .data_transformation import datatransformation_class
from .data_transformation import datatranformation_preprocess

from .model_training import model_train_path
from .model_training import model_trainer



@dataclass
class datainjectionconfig:
    train_path :str=os.path.join('artifact','train.csv')
    test_path :str=os.path.join('artifact','test.csv')
    raw_path :str=os.path.join('artifact','data.csv')

class datainjection:
    def __init__(self):
            self.injectionconfig=datainjectionconfig()
        
    def initiate_injection(self):
        try:
            logging.info('Entered into logging block')
            df=pd.read_csv(r'D:\Mlops\data\StudentsPerformance.csv')

            logging.info('Reading the dataset')
            os.makedirs(os.path.dirname(self.injectionconfig.train_path), exist_ok=True)

            df.to_csv(self.injectionconfig.raw_path,index=False,header=True)

            logging.info('Train test split initiated')
            train_set,test_set=train_test_split(df,test_size=0.25,random_state=44)

            train_set.to_csv(self.injectionconfig.train_path,index=False,header=True)

            test_set.to_csv(self.injectionconfig.test_path,index=False,header=True)
            logging.info('injection completed')
            return(
                 self.injectionconfig.train_path,
                 self.injectionconfig.test_path
            )

        except Exception as e:
             raise CustomeException(e,sys)
        
if __name__=="__main__":
     obj=datainjection()
     train_data, test_data=obj.initiate_injection()

     datatransformation=datatranformation_preprocess()
     train_arr,test_arr,_=datatransformation.initiate_data_transformation(train_data,test_data)

     model_training=model_trainer()
     model_training.initiate_model_training(train_arr,test_arr)


            
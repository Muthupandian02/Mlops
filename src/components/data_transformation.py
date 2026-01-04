import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomeException

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import savefile

@dataclass
class datatransformation_class:
    preprocessor_file=os.path.join('artifact','preprocessor.pkl')

class datatranformation_preprocess:
    def __init__(self):
        self.preprocess_path=datatransformation_class()

    def datatransformation_convert(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                    "gender",
                    "race_or_ethnicity",
                    "parental_level_of_education",
                    "lunch",
                    "test_preparation_course",
                ]
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                    ]
                )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('onehot',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                    ]
                )
            logging.info('num_pipeline,cat_pipeline created')
            preprocessor=ColumnTransformer([
                ('numeric',num_pipeline,numerical_columns),
                ('categorical',cat_pipeline,categorical_columns)
            ])
            logging.info('preprocessor created')
            return preprocessor
            
        except Exception as e:
            raise CustomeException(e,sys)
            
    def initiate_data_transformation(self,train_Path,test_Path):
        try:
            train_data=pd.read_csv(train_Path)
            test_data=pd.read_csv(test_Path)
            logging.info('train and test path are readed')

            preprocessor=self.datatransformation_convert()

            target_column="math_score"
            numeric_columns=['reading_score','writing_score']


            train_set=train_data.drop(columns=[target_column],axis=1)
            target_train_input=train_data[target_column]

            test_set=test_data.drop(columns=[target_column],axis=1)
            target_test_input=test_data[target_column]

            logging.info('train and target_train, test and target_test were created')

            train_enc=preprocessor.fit_transform(train_set)
            test_enc=preprocessor.transform(test_set)

            logging.info('train and test are encoded')

            train_arr=np.c_[train_enc,np.array(target_train_input)]
            test_arr=np.c_[test_enc,np.array(target_test_input)]

            savefile(
                file_path=self.preprocess_path.preprocessor_file,
                obj=preprocessor
            )
            
            logging.info('saving the file initiated')
            logging.info('train_array and test_array were created')
            logging.info('data transformation were done')
            return (train_arr,
                    test_arr,
                    self.preprocess_path.preprocessor_file)
        except Exception as e:
            raise CustomeException(e,sys)
    

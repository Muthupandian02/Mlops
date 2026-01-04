import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomeException
from src.utils import load_file

class predictpipeline:
    def __init__(self):
        pass
    
    def prediction(self,features):
        try:
            model_path= os.path.join('artifact','model.pkl')
            preprocess_path=os.path.join('artifact','preprocessor.pkl')
            model=load_file(model_path)
            preprocessor=load_file(preprocess_path)
            preprocessed_data=preprocessor.transform(features)
            predicted_op=model.predict(preprocessed_data)
            return predicted_op
        except Exception as e:
            raise CustomeException(e,sys)
        
class user_input_data:
    def __init__(
            self,
            gender: str,
            race_or_ethnicity: str,
            parental_level_of_education: str,
            lunch: str,
            test_preparation_course: str,
            math_score: int,
            reading_score: int,
            writing_score: int):
        self.gender=gender,
        self.race_or_ethnicity=race_or_ethnicity,
        self.parental_level_of_education=parental_level_of_education,
        self.lunch=lunch,
        self.test_preparation_course=test_preparation_course,
        self.math_score=math_score,
        self.reading_score=reading_score,
        self.writing_score=writing_score
    def getting_data(self):
        try:
           user_input={
            'gender':[self.gender],
            'race_or_ethnicity':[self.race_or_ethnicity],
            'parental_level_of_education':[self.parental_level_of_education],
            'genlunchder':[self.lunch],
            'test_preparation_course':[self.test_preparation_course],
            'math_score':[self.math_score],
            'reading_score':[self.reading_score],
            'writing_score':[self.writing_score]
            }
           return pd.DataFrame(user_input)
    
        except Exception as e:
            raise CustomeException(e,sys)
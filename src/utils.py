import os
import sys
import dill
import pickle

from src.exception import CustomeException
from src.logger import logging
from sklearn.metrics import r2_score

def savefile(file_path,obj):
    try:
        dirname=os.path.dirname(file_path)

        os.makedirs(dirname,exist_ok=True)

        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
        
        logging.info('file was dumped')
    except Exception as e:
        raise CustomeException(e,sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        logging.info('Model evaluation entered')
        report={}
        for i in range(len(list(models))):
            model=list((models.values()))[i]
            
            model.fit(x_train,y_train)
            prediction1=model.predict(x_train)
            prediction2=model.predict(x_test)

            train_model_score=r2_score(y_train,prediction1)
            test_model_score=r2_score(y_test,prediction2)

            report[list(models.keys())[i]]=test_model_score

        return report
    
    except Exception as e:
        raise CustomeException(e,sys)
    
def load_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
        
    except Exception as e:
        raise CustomeException(e,sys)
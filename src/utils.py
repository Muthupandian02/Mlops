import os
import sys
import dill
import pickle

from src.exception import CustomeException
from src.logger import logging

def savefile(file_path,obj):
    try:
        dirname=os.path.dirname(file_path)

        os.makedirs(dirname,exist_ok=True)

        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
        
        logging.info('file waas dumped')
    except Exception as e:
        raise CustomeException(e,sys)
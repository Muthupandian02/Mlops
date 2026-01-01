import sys
import os
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.utils import evaluate_model
from src.logger import logging
from src.exception import CustomeException
from src.utils import savefile
from sklearn.metrics import r2_score


@dataclass
class model_train_path:
    model_file=os.path.join('artifact','model.pkl')

class model_trainer:
    def __init__(self):
        self.model_path=model_train_path()
    
    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Block: model training entered')
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                'RandomForestRegressor':RandomForestRegressor(),
                'AdaBoostRegressor':AdaBoostRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor(),
                'LinearRegression':LinearRegression(),
                'KNeighborsRegressor':KNeighborsRegressor(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'XGBRegressor':XGBRegressor()
            }
            model_report:dict=evaluate_model(
                x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models
                )
            logging.info('model get trained and predicted')

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomeException('No best model found')
            logging.info('Best model found')

            savefile(
                file_path=self.model_path.model_file,
                obj=best_model
            )
            prediction=best_model.predict(x_test)

            r2_score_=r2_score(y_test,prediction)
            logging.info('Model training finished!')
            return r2_score_

        except Exception as e:
            raise CustomeException(e,sys)


import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

class DataStrategy(ABC):
    def handle_data(self, data:pd.DataFrame) ->Union[pd.DataFrame,pd.Series]:
        pass



class Data_pre_process_strategy(DataStrategy):
    def handle_data(self, data:pd.DataFrame) ->pd.DataFrame:
        try:
            data=data.drop(columns="company",axis=1,inplace=True)
            data=data.dropna(axis=0)
            data=data.drop_duplicates()
            return data

        except Exception as err:
            logging.error("error in preprocessing data {err}")
            raise err
    
    def get_data_transformer_object(self,data:pd.DataFrame) -> pd.DataFrame:
        try:
            numerical_columns=data.select_dtypes(include=[np.number])
            categorical_columns = data.select_dtypes(include=["object"])

            num_pipeline= Pipeline(
                steps=[
                ("scaler",StandardScaler())

                ]
            )

            categorical_pipeline=Pipeline(

                steps=[
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",categorical_pipeline,categorical_columns)

                ]
            )

            return preprocessor
        
        except Exception as err:
            raise err

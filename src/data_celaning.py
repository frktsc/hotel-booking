import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data, transformed_data, preprocessor = self.preprocess_and_transform(data)
            return transformed_data, preprocessor, data


        except Exception as err:
            logging.error(f"Error in processing data: {err}")
            raise

    def preprocess_and_transform(self, data: pd.DataFrame) -> (pd.DataFrame, Pipeline):
        try: 
            data = data.drop(columns=["company"], axis=1)
            data = data.dropna(axis=0)
            data = data.drop_duplicates()

            numerical_columns = data.select_dtypes(include=[np.number]).columns
            categorical_columns = data.select_dtypes(include=["object"]).columns

            num_pipeline = Pipeline([("scaler", StandardScaler())])
            cat_pipeline = Pipeline([("one_hot_encoder", OneHotEncoder()),
                                     ("scaler", StandardScaler(with_mean=False))])
            
            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, numerical_columns),
                 ("cat_pipeline", cat_pipeline, categorical_columns)]
            )
            transformed_data = preprocessor.fit_transform(data)
            return data, transformed_data, preprocessor

        except Exception as err:
            logging.error(f"Error in processing data: {err}")
            raise

class DatadivedStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame ,pd.Series]:
        try:
            X=data.drop(["is_canceled"],axis=1)
            y=data["is_canceled"]
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as err:
            logging.error(f"Error in dividing data {err}")
            raise

class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.df)
        except Exception as err:
            raise   
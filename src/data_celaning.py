import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(columns=["company"], axis=1)
            data = data.dropna(axis=0)
            data = data.drop_duplicates()

            data = data.select_dtypes(include=[np.number])
            categorical_columns = data.select_dtypes(include=["object"]).columns
            data = data.drop(categorical_columns, axis=1)

            return data
        except Exception as err:
            logging.error(err)
            raise



class DataDivideStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:

        try:
            X = data.drop("is_canceled", axis=1)
            y = data["is_canceled"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        
        except Exception as err:
            logging.error(f"error happned{err}")
            raise


class DataCleaning:

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        return self.strategy.handle_data(self.df)
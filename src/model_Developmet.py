from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
import logging

class model(ABC):

    @abstractmethod
    def train(self, x_train, y_train):
        pass

class Logisticregressionmodel(model):
    
    def train(self, x_train, y_train,**kwargs):
        try:
            lor = LogisticRegression(**kwargs)
            lor.fit(x_train,y_train)
            logging.info("Model training completed")
            return lor
        except Exception as err:
            logging.error(f"error in training model: {err}")
            raise
    





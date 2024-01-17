import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix


class Evaluation(ABC):

    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray, y_pred:np.ndarray):
        pass

class ACC(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating ACCURACY")
            acc = accuracy_score(y_true, y_pred)
            logging.info(f"acc: {acc}")
            return acc
        except Exception as err:
            logging.error(f"error in calculating {acc}")
            raise

class CFM(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating Confusion Matrix")
            cfm = confusion_matrix(y_true,y_pred)
            logging.info(f"CFM : {cfm}")
            return cfm
        except Exception as err:
            logging.error(f"error in calculating {err}")
            raise
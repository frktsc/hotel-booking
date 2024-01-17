import logging
import pandas as pd
from zenml import step
from src.evaluation import ACC,CFM
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
from typing  import Tuple
import numpy as np
from zenml.client import Client
import mlflow


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def Evaluation_Data(model: ClassifierMixin,
                    X_test: pd.DataFrame,
                    y_test: pd.Series) -> Tuple[Annotated[float, "ACC"],
                                              Annotated[np.ndarray, "CFM"]]:
    try:
        prediction = model.predict(X_test)
        acc_class = ACC()
        acc = acc_class.calculate_scores(y_test.values, prediction)
        mlflow.log_metric("acc", acc)

        cfm_class = CFM()
        cfm = cfm_class.calculate_scores(y_test.values, prediction)
        cfm_acc = np.diag(cfm).sum() / cfm.sum()
        mlflow.log_metric("accuracy", cfm_acc)
        return acc, cfm
    except Exception as err:
        logging.error(f"Error in evaluating model: {err}")
        raise


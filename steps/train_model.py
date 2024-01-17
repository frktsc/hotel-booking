import logging
import pandas as pd
from zenml import step
from src.model_Developmet import Logisticregressionmodel
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client


experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def data_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig
) -> ClassifierMixin:

    model = None
    if config.model_name == "LogisticRegression":
        mlflow.autolog()
        model = Logisticregressionmodel()
        trained_model = model.train(X_train,y_train)
        return trained_model
    else:
        raise ValueError(f"model {config.model_name} not supported")

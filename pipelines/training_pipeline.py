from zenml import pipeline
from steps.cleaning_Data import clean_Data
from steps.evaluation_Data import Evaluation_Data
from steps.Ingest_Data import ingest_df
from steps.train_model import data_train
from steps.config import ModelNameConfig

model_config = ModelNameConfig(model_name="LogisticRegression")

@pipeline(enable_cache=True)
def Train_Pipeline(data_path:str):
    df=ingest_df(data_path)
    X_train, X_test, y_train, y_test=clean_Data(df)
    model = data_train(X_train, X_test, y_train, y_test,model_config)
    acc, cfm = Evaluation_Data(model, X_test, y_test)
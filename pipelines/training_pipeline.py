from zenml import pipeline
from steps.cleaning_Data import clean_Data
from steps.evaluation_Data import Evaluation_Data
from steps.Ingest_Data import ingest_df
from steps.train_model import data_train

@pipeline(enable_cache=True)
def Train_Pipeline(data_path:str):
    df=ingest_df(data_path)
    clean_Data(df)
    data_train(df)
    Evaluation_Data(df)

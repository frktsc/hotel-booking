import logging
import pandas as pd
from zenml import step

class ingestData():
    def __init__(self, data_path):
        self.data_path=data_path
        

    def get_data(self):
        logging.info(f"ingested data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_df(data_path:str) -> pd.DataFrame:   
    try:
        ingest_Data=ingestData(data_path)
        df = ingest_Data.get_data()
        return df
    
    except Exception as err:
        logging.error(f"error while ingesting data {err}")
        raise err
    

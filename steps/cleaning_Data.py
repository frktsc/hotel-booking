import logging
import pandas as pd
from zenml import step
from src.data_celaning import DataPreprocessStrategy, DataCleaning, DataDivideStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_Data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],
]:
    try:
        process_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_Data=data_cleaning.handle_data()
        divide_Strategy=DataDivideStrategy()
        data_cleaning=DataCleaning(processed_Data,divide_Strategy)
        X_train, X_test, y_train, y_test =data_cleaning.handle_data()
        logging.info("data cleaning completed")
        return X_train, X_test, y_train, y_test
    except Exception as err:
        logging.error(f"error in clenain data {err}")
        raise

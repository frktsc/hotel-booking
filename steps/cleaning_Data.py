import logging
import pandas as pd
from zenml import step

@step
def clean_Data(df: pd.DataFrame) -> None:
    pass

import logging
import pandas as pd
from zenml import step


@step
def data_train(df: pd.DataFrame) -> None:

    pass
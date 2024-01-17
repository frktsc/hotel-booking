import logging

import pandas as pd
from src.data_celaning import DataCleaning, DataPreprocessStrategy


def get_data_for_test():
    try:
        df = pd.read_csv(r"C:\Users\furkanbaba\Desktop\furkanbaba\coding\hotel-booking\data\data_hotel_booking.csv")
        df = df.sample(n=100)
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["is_canceled"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as err:
        logging.error(err)
        raise
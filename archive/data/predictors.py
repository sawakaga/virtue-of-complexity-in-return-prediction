# put data on dataframe
# clean data
# preprocess data (dont need for now)
# return data

import pandas as pd

class GoyalPredictors:
    def __init__(self, data):
        self.data = pd.read_csv(data)

    def clean_data(self):
        return

    def preprocess_data(self):
        return

    def get_data(self):
        return self.data

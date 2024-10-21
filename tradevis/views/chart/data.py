import dask.dataframe as dd


DEFAULT_PATH = 'resources/training_data.csv'

def _load_data(path: str = DEFAULT_PATH):
    data = dd.read_csv(path)
    data = data.compute()
    data = data[["bid_open", "bid_close", "bid_high", "bid_low", "EMA_21", "EMA_50", "EMA_200"]]
    return data


data = _load_data()

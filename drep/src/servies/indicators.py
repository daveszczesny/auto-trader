import pandas as pd
from src.servies.logger import logger

CLOSE = 'ask_close'

def csv_to_dataframe(file: str) -> pd.DataFrame:
    """
    Read csv file and return as DataFrame
    """
    try:
        data = pd.read_csv(file)

    except Exception as e:
        logger.error(f"Failed to read {file}. Error: {e}")
        raise e

    return data


def dataframe_to_csv(data: pd.DataFrame, file: str):
    """
    Write DataFrame to csv file
    """
    try:
        data.to_csv(file, index=False)

    except Exception as e:
        logger.error(f"Failed to write to {file}. Error: {e}")
        raise e


def simple_moving_average(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate simple moving average
    """
    data['SMA'] = data[CLOSE].rolling(window=window).mean()
    return data


def exponential_moving_average(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate exponential moving average
    """
    data['EMA'] = data[CLOSE].ewm(span=window, adjust=False).mean()
    return data


def relative_strength_index(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate relative strength index
    """
    delta = data[CLOSE].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))

    data['RSI'] = RSI
    return data


def average_true_range(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate average true range
    """
    data['high_low'] = data['high'] - data['low']
    data['high_close'] = abs(data['high'] - data[CLOSE].shift())
    data['low_close'] = abs(data['low'] - data[CLOSE].shift())

    data['true_range'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
    data['ATR'] = data['true_range'].rolling(window=window).mean()
    
    return data


# volume not in data
def volume_weighted_average_price(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate volume weighted average price
    """
    data['volume'] = data[['ask_volume_sum', 'bid_volume_sum']].sum(axis=1)
    data['VWAP'] = (data[CLOSE] * data['volume']).cumsum() / data['volume'].cumsum()

    return data
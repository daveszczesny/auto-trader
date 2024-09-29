import os
import pandas as pd
from drep.src.servies.logger import logger

CLOSE = 'ask_close'

def csv_to_dataframe(file: str) -> pd.DataFrame:
    """
    Read csv file and return as DataFrame
    """

    os.chdir('../resources/')

    try:
        data = pd.read_csv(file)

    except Exception as e: # pylint: disable=broad-except
        logger.log_error(f"Failed to read {file}. Error: {e}")
        raise e

    return data


def dataframe_to_csv(data: pd.DataFrame, file: str):
    """
    Write DataFrame to csv file
    """
    try:
        data.to_csv(file, index=False)

    except Exception as e: # pylint: disable=broad-except
        logger.log_error(f"Failed to write to {file}. Error: {e}")
        raise e


def add_indicator(data: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """
    Add indicator to dataset
    """
    try:
        if indicator.startswith('SMA'):
            window = int(indicator.split('_')[1])
            data = simple_moving_average(data, window)

        elif indicator.startswith('EMA'):
            window = int(indicator.split('_')[1])
            data = exponential_moving_average(data, window)

        elif indicator == 'RSI':
            data = relative_strength_index(data)

        elif indicator == 'ATR':
            data = average_true_range(data)

        elif indicator == 'VWAP':
            data = volume_weighted_average_price(data)

    except Exception as e: # pylint: disable=broad-except
        logger.log_error(f"Failed to add {indicator} to dataset. Error: {e}")
        raise e

    return data


def remove_indicator(data: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """
    Remove indicator from dataset
    """
    try:
        data.drop('EMA', axis=1, inplace=True)

    except Exception as e: # pylint: disable=broad-except
        logger.log_error(f"Failed to remove {indicator} from dataset. Error: {e}")

    return data


def simple_moving_average(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate simple moving average
    """
    data[f'SMA_{window}'] = data[CLOSE].rolling(window=window).mean()
    return data


def exponential_moving_average(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate exponential moving average
    """
    data[f'EMA_{window}'] = data[CLOSE].ewm(span=window, adjust=False).mean()
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


def volume_weighted_average_price(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate volume weighted average price
    """
    data['volume'] = data[['ask_volume_sum', 'bid_volume_sum']].sum(axis=1)
    data['VWAP'] = (data[CLOSE] * data['volume']).cumsum() / data['volume'].cumsum()

    return data

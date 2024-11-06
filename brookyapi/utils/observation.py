import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np

from utils.exceptions import ErrorSet
from utils.constants import StatusCode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BrookyAPI")

def get_observation(payload: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[ErrorSet], int]:
    """
    Construct an observation from the payload.

    :param payload: 
        The input data containing 
            - balance, 
            - unrealized_pnl,
            - current_price,
            - current_high,
            - current_low,
            - open_trades,
            - indicators
        :return: A tuple containing the observation array and 
            an error message if any required fields are missing.
    """
    balance = payload.get('balance', None)
    unrealized_pnl = payload.get('unrealized_pnl', None)
    current_price = payload.get('current_price', None)
    current_high = payload.get('current_high', None)
    current_low = payload.get('current_low', None)
    open_trades = payload.get('open_trades', None)
    indicators = payload.get('indicators', None)

    if any(value is None for value in (balance,
                                       unrealized_pnl,
                                       current_price,
                                       current_high,
                                       current_low,
                                       open_trades,
                                       indicators)):
        logger.error(f'Required fields not found in payload: {repr(payload)}')
        return None, ErrorSet.INVALID_INPUT, StatusCode.BAD_REQUEST

    # retrieve expected indicators from payload
    indicator_values = {indicator['name']: indicator['value'] for indicator in indicators}
    ema_200 = indicator_values.get('ema_200', None)
    ema_50 = indicator_values.get('ema_50', None)
    ema_21 = indicator_values.get('ema_21', None)

    if any(value is None for value in (ema_200, ema_50, ema_21)):
        logger.error(f'Expected indicators not found in payload: {repr(indicator_values)}')
        return None, ErrorSet.INVALID_INPUT, StatusCode.BAD_REQUEST

    # construct observation
    observation = np.array(
        [
            balance,
            unrealized_pnl,
            current_price,
            current_high,
            current_low,
            open_trades,
            ema_200,
            ema_50,
            ema_21
        ]
    )


    if len(observation) != 9:
        # This should never happen
        logger.error(f'Observation does not contain 9 elements: {repr(observation)}')
        return ErrorSet.INVALID_OBSERVATION_LENGTH, StatusCode.BAD_REQUEST

    return observation, None, StatusCode.OK



def get_observation_list(payload: Dict[str, Any]):

    logger.info('Processing observation list')
    balance = payload.get('balance', None)
    prices = payload.get('current_prices', None)
    highs = payload.get('current_highs', None)
    lows = payload.get('current_lows', None)
    indicators = payload.get('indicators', None)

    if any(value is None for value in (balance,
                                       prices,
                                       highs,
                                       lows,
                                       indicators)):
        logger.error(f'Required fields not found in payload: {repr(payload)}')
        return None, ErrorSet.INVALID_INPUT, StatusCode.BAD_REQUEST

    indicator_values = {indicator['name']: indicator['value'] for indicator in indicators}
    ema_200 = indicator_values.get('ema_200', None)
    ema_50 = indicator_values.get('ema_50', None)
    ema_21 = indicator_values.get('ema_21', None)

    if any(value is None for value in (ema_200, ema_50, ema_21)):
        logger.error(f'Expected indicators not found in payload: {repr(indicator_values)}')
        return None, ErrorSet.INVALID_INPUT, StatusCode.BAD_REQUEST

    if len(prices) != len(highs) or \
        len(prices) != len(lows) or \
        len(prices) != len(ema_200) or \
        len(prices) != len(ema_50) or \
        len(prices) != len(ema_21):
        logger.error(f'Invalid input: Length of prices "{len(prices)}'
                     f'highs "{len(highs)}", lows "{len(lows)}" '
                     f'ema200 "{len(ema_200)}", ema50 "{len(ema_50)}" and'
                     f'ema21 "{len(ema_21)}" do not match')
        return None, ErrorSet.INVALID_INPUT, StatusCode.BAD_REQUEST

    observation_list = []
    for i, prices in enumerate(prices):
        observation = np.array(
            [
                balance,
                0.0, # unrealized_pnl
                prices,
                highs[i],
                lows[i],
                0, # open_trades
                ema_200[i],
                ema_50[i],
                ema_21[i]
            ]
        )
        observation_list.append(observation)

    logger.info('Observation list processed successfully')
    return observation_list, None, StatusCode.OK

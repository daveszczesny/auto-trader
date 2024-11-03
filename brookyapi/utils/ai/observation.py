import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np

from utils.ai.exceptions import ErrorSet, StatusCode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BrookyAPI")

def construct_observation(payload: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[ErrorSet], int]:
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
        logger.error(f'Observation does not contain 9 elements: {repr(observation)}')
        return ErrorSet.INVALID_OBSERVATION_LENGTH, StatusCode.BAD_REQUEST

    return observation, None, StatusCode.OK

import logging
from typing import Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BrookyAPI")

def construct_observation(payload) -> Tuple[np.ndarray, Optional[str]]:
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
        logger.error(f'Missing required fields: {repr(payload)}')
        return None, 'Missing required fields'
    
    # retrieve expected indicators from payload
    indicator_values = {indicator['name']: indicator['value'] for indicator in indicators}
    ema_200 = indicator_values.get('ema_200', None)
    ema_50 = indicator_values.get('ema_50', None)
    ema_21 = indicator_values.get('ema_21', None)

    if any(value is None for value in (ema_200, ema_50, ema_21)):
        logger.error(f'Missing expected indicators: {repr(indicator_values)}')
        return None, 'Missing expected indicators'
    

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
        return 'Observation does not contain 9 elements'
    
    return observation, None
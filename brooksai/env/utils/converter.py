from brooksai.env.models.constants import ApplicationConstants

def pip_to_profit(pip_delta: float, lot_size: float) -> float:
    """
    Convert pips to profit
    """
    return pip_delta * lot_size * ApplicationConstants.CONTRACT_SIZE


def pips_to_price_chart(pips: float) -> float:
    """
    Convert pips to price chart
    """
    return pips / 10_000

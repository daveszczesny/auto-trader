from brooksai.env.models.constants import CONTRACT_SIZE

def pip_to_profit(pip_delta: float, lot_size: float) -> float:
    """
    Convert pips to profit
    """
    return pip_delta * lot_size * CONTRACT_SIZE


def pips_to_price_chart(pips: float) -> float:
    """
    Convert pips to price chart
    """
    return pips / 10_000

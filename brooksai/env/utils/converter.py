
from currency_converter import CurrencyConverter
c = CurrencyConverter()

def pip_to_profit(pip_delta: float, lot_size: float) -> float:
    """
    Convert pips to profit
    """
    return c.convert(pip_delta * lot_size * 100_000, "USD", "GBP")


def pips_to_price_chart(pips: float) -> float:
    """
    Convert pips to price chart
    """
    return pips / 10_000
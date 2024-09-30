from uuid import uuid1
from typing import Optional, Union, List, Dict

from currency_converter import CurrencyConverter

from brooksai.env.models.constants import TradeType, Environments,\
    DEFAULT_TRADE_TTL, ENVIRONMENT, CONTRACT_SIZE, LEVERAGE
from brooksai.env.utils.converter import pip_to_profit, pips_to_price_chart

c = CurrencyConverter()

class Trade:
    """
    The trade class represents one trade
    """

    ttl: int = DEFAULT_TRADE_TTL # 10 days
    _lot_size: float
    _open_price: float
    _trade_type: TradeType
    stop_loss: Optional[float]
    take_profit: Optional[float]

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(self,
                 lot_size: float,
                 open_price: float,
                 trade_type: TradeType,
                 stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None):
        self.uuid = str(uuid1())
        self._lot_size = round(lot_size if lot_size > 0 else 0.01, 2)
        self._open_price = open_price
        self.trade_type = trade_type
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        if self.stop_loss and self.stop_loss < 0:
            self.stop_loss = None
        if self.take_profit and self.take_profit < 0:
            self.take_profit = None

        if ENVIRONMENT == Environments.DEV:
            with open('brooksai_logs.txt', 'a') as f:
                f.write(f"Trade {self.uuid} opened with lot size: {self._lot_size},"
                        f"open price: {open_price}, trade type: {trade_type},"
                        f"SL: {self.stop_loss}, TP: {self.take_profit}\n")
        open_trades.append(self)

    @property
    def lot_size(self) -> float:
        return self._lot_size

    @lot_size.setter
    def lot_size(self, value: float) -> None:
        if value < 0:
            value = 0.01
        self._lot_size = value

    @property
    def trade_type(self) -> TradeType:
        return self._trade_type

    @trade_type.setter
    def trade_type(self, value: TradeType) -> None:
        self._trade_type = value

    @property
    def open_price(self) -> float:
        return self._open_price

    def get_margin(self) -> float:
        """
        Calculate the margin
        """
        # Formula for pip to cost
        # lot size * contract size * EUR TO GBP
        return ((self.lot_size * CONTRACT_SIZE) / LEVERAGE) * c.convert(1, 'EUR', 'GBP')

def check_margin(lot_size: float) -> float:
    """
    Check if the margin is enough to open a trade
    """
    return ((lot_size * CONTRACT_SIZE) / LEVERAGE) * c.convert(1, 'EUR', 'GBP')


def reset_open_trades():
    global open_trades
    open_trades.clear()

def get_trade_by_id(uuid: str) -> Optional[Trade]:
    for trade in open_trades:
        if trade.uuid == uuid:
            return trade
    return None

def close_all_trades(current_price: float) -> float:
    global open_trades
    total_value: float = 0.0
    while len(open_trades) > 0:
        total_value += close_trade(open_trades[0], current_price=current_price)
    return total_value

def get_trade_profit(trade: Trade, current_price: float) -> float:
    trade_profit_in_pips: float = 0.0
    if trade.trade_type == TradeType.LONG:
        trade_profit_in_pips = current_price - trade.open_price
    else:
        trade_profit_in_pips = trade.open_price - current_price

    return pip_to_profit(trade_profit_in_pips, trade.lot_size) * c.convert(1, 'USD', 'GBP')

def get_trade_state(uuid: str, current_price: float) -> Dict[str, Union[str, float]]:
    trade = get_trade_by_id(uuid)

    trade_state = get_trade_profit(trade, current_price)

    return {
        "id": trade.uuid,
        "lot_size": trade.lot_size,
        "trade_type": trade.trade_type,
        "profit": trade_state
    }


def trigger_stop_or_take_profit(current_price: float) -> float:

    """
    This method checks if the current price has hit the stop loss or take profit of a trade

    This method is not necessary when using prod environment since the broker will handle this

    :param current_price: The current price of the asset
    :return: The total value of all trades closed by the stop loss or take profit
    """

    if ENVIRONMENT == Environments.PROD:
        return 0.0

    global open_trades

    total_value: float = 0.0
    for trade in open_trades:
        # Check if take profit or stop loss is hit
        # stop loss and take profit are in pips so we must convert them to the price chart
        # add or subtract them from open price
        if trade.trade_type == TradeType.LONG:
            take_profit_price = trade.open_price +\
                pips_to_price_chart(trade.take_profit) if trade.take_profit else None
            stop_loss_price = trade.open_price -\
                pips_to_price_chart(trade.stop_loss) if trade.stop_loss else None

            if take_profit_price and current_price >= take_profit_price:
                total_value += close_trade(trade, take_profit_price)
            elif stop_loss_price and current_price <= stop_loss_price:
                total_value += close_trade(trade, stop_loss_price)

        elif trade.trade_type == TradeType.SHORT:
            take_profit_price = trade.open_price -\
                pips_to_price_chart(trade.take_profit) if trade.take_profit else None
            stop_loss_price = trade.open_price +\
                pips_to_price_chart(trade.stop_loss) if trade.stop_loss else None

            if take_profit_price and current_price <= take_profit_price:
                total_value += close_trade(trade, take_profit_price)
            elif stop_loss_price  and current_price >= stop_loss_price:
                total_value += close_trade(trade, stop_loss_price)

    return total_value

def open_trade(trade: Trade):
    Trade(
        lot_size=trade.lot_size,
        open_price=trade.open_price,
        trade_type=trade.trade_type,
        stop_loss=trade.stop_loss,
        take_profit=trade.take_profit
    )

def close_trade(trade: Trade, current_price: Optional[float]) -> float:

    if trade not in open_trades:
        # Trade was already closed before agent could close it
        return 0.0

    value: float = get_trade_profit(trade, current_price)
    open_trades.remove(trade)

    if ENVIRONMENT != Environments.PROD:
        with open('brooksai_logs.txt', 'a') as f:
            f.write(f"Trade {trade.uuid}, TTL: {trade.ttl} closed with profit: {round(value, 2)}\n")

    # TODO: Send request to broker to close trade

    return value

open_trades: List[Trade] = []

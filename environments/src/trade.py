
from typing import Optional
from currency_converter import CurrencyConverter
from uuid import uuid4

from gamemanager import game_manager
from agent import Agent
from utils.constants import TradeType

c = CurrencyConverter()

class Trade:
    """
    The trade class represents one trade
    """

    uuid: str = str(uuid4())
    in_trade: bool = False
    lot_size: float = 0.01
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    price_at_enter: Optional[float] = None
    type_of_trade: Optional[TradeType] = None
    trade_livespan_in_minutes: int = 0
    agent: Optional[Agent] = None

    MARGIN_CALL: float = 0.4
    COMMISSION_RATE: float = 2.54

    def __init__(self, agent: Agent) -> None:
        self.agent = agent


    def long(self, lots: float, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool:
        """
        Go long on a currency pair
        """
        if self.in_trade:
            return False

        self.type_of_trade = TradeType.LONG
        return self._enter_trade(lots, stop_loss, take_profit)

    def short(self, lots: float, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool:
        """
        Go short on a currency pair
        """
        if self.in_trade:
            return False
        
        self.type_of_trade = TradeType.SHORT
        return self._enter_trade(lots, stop_loss, take_profit)

    def close_trade(self) -> float:
        """
        Close the trade
        """
        profit: float = self._close_buy_trade() if TradeType.LONG else self._close_sell_trade()
        profit -= self._trade_commission()

        return round(profit, 2)


    def get_cost_to_enter_trade(self) -> float:
        """
        Get the cost to enter the trade
        """
        current_price: float = game_manager.get_current_ask_price() if self.type_of_trade == TradeType.LONG else game_manager.get_current_bid_price()
        price_in_pips: float = current_price * 10_000
        price_in_usd: float = price_in_pips * 10
        leveraged_price: float = price_in_usd / game_manager.leverage
        price_for_lots: float = leveraged_price * self.lot_size
        return price_for_lots * c.convert(1, 'USD', 'GBP')

    def get_state_of_trade(self) -> float:
        """
        Get the state of the trade
        """
        current_price: float = game_manager.get_current_ask_price() if self.type_of_trade == TradeType.LONG else game_manager.get_current_bid_price()
        return self._calculate_profit(current_price)


    def _enter_trade(self, lots: float, stop_loss: float | None = None, take_profit: float | None = None) -> bool:
        """
        Enter a trade
        """

        if not self._assert_lot_size(lots) or not self._assert_trade():
            return False

        self.lot_size = lots
        self.in_trade = True
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.price_at_enter = game_manager.get_current_ask_price() if self.type_of_trade == TradeType.LONG else game_manager.get_current_bid_price()
        return True

    def _trade_commission(self) -> float:
        """
        Calculate the trade commission
        """
        return self.lot_size * self.COMMISSION_RATE

    def _assert_trade(self) -> bool:
        """
        Check if the agent has enough balance to enter a trade,
        Agent must have at least 40% of the cost to enter the trade
        This is to ensure the agent has enough balance to cover margin calls (50%)
        """
        return False if self.agent.account_balance * self.MARGIN_CALL < self.get_cost_to_enter_trade() else True

    def _assert_lot_size(self, value: float) -> bool:
        """
        Check if the lot size is valid
        """
        return value >= 0.01

    def _close_buy_trade(self) -> float:
        """
        Close the long trade
        """
        bid_price: float = game_manager.get_current_bid_price()
        return self._calculate_profit(bid_price)
    
    def _close_sell_trade(self) -> float:
        """
        Close the short trade
        """
        ask_price: float = game_manager.get_current_ask_price()
        return self._calculate_profit(ask_price)
    
    def _calculate_profit(self, current_price: float) -> float:
        """
        Calculate the profit of the trade
        """
        # Pips difference
        pip_delta: float = self.price_at_enter - current_price
        price_in_pips: float = pip_delta * 10_000
        profit: float = self._convert_pip_to_usd(price_in_pips) * self.lot_size * c.convert(1, 'USD', 'GBP')
        return profit

    def _convert_pip_to_usd(self, pip: float) -> float:
        """
        Convert pips to USD
        """
        return pip * 10


from typing import Optional
from currency_converter import CurrencyConverter

from gamemanager import game_manager
from agent import Agent
from utils.constants import TradeType

c = CurrencyConverter()

class Trade:
    """
    The trade class represents one trade
    """

    in_trade: bool = False
    lots: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    price_at_enter: float | None = None
    type_of_trade: TradeType | None = None

    agent: Agent | None= None



    def __init__(self, agent: Agent) -> None:
        self.agent = agent


    def long(self, lots: float, stop_loss: float | None = None, take_profit: float | None = None) -> bool:
        """
        Go long on a currency pair
        """
        self.type_of_trade = TradeType.LONG
        self.lots = lots

        if not self.assert_trade(): return False

        self.in_trade = True
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.price_at_enter = self.get_current_ask_price()

        return True

    def short(self, lots: float, stop_loss: float | None = None, take_profit: float | None = None) -> bool:
        """
        Go short on a currency pair
        """
        self.type_of_trade = TradeType.SHORT
        self.lots = lots

        if not self.assert_trade(): return False

        self.in_trade = True
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.price_at_enter = self.get_current_bid_price()

        return True

    def close_trade(self) -> float:
        """
        Close the trade
        """

        profit: float = 0.0

        if self.type_of_trade == TradeType.LONG:
            profit = self._close_buy_trade()
        elif self.type_of_trade == TradeType.SHORT:
            profit = self._close_sell_trade()

        profit -= self.trade_commission()

        return round(profit, 2)

    def trade_commission(self) -> float:
        """
        Calculate the trade commission
        """
        return self.lots * 2.54

    def get_cost_to_enter_trade(self) -> float:
        """
        Get the cost to enter the trade
        """
        current_price: float = self.get_current_ask_price() if self.type_of_trade == TradeType.LONG else self.get_current_bid_price()
        price_in_pips: float = current_price * 10_000
        price_in_usd: float = price_in_pips * 10
        leveraged_price: float = price_in_usd / game_manager.leverage
        price_for_lots: float = leveraged_price * self.lots
        return price_for_lots * c.convert(1, 'USD', 'GBP')

    def assert_trade(self) -> bool:
        """
        Check if the agent has enough balance to enter a trade,
        Agent must have at least 40% of the cost to enter the trade
        This is to ensure the agent has enough balance to cover margin calls (50%)
        """

        if self.agent.account_balance * 0.5 < self.get_cost_to_enter_trade():
            return False
        
        return True

    def _close_buy_trade(self) -> float:
        """
        Close the long trade
        """
        bid_price: float = self.get_current_bid_price()
        return self._calculate_profit(bid_price)
    
    def _close_sell_trade(self) -> float:
        """
        Close the short trade
        """
        ask_price: float = self.get_current_ask_price()
        return self._calculate_profit(ask_price)
    
    def _calculate_profit(self, current_price: float) -> float:
        """
        Calculate the profit of the trade
        """
        # Pips difference
        pip_delta: float = self.price_at_enter - current_price
        price_in_pips: float = pip_delta * 10_000
        profit: float = self.convert_pip_to_usd(price_in_pips) * self.lots * c.convert(1, 'USD', 'GBP')

        return profit

    def convert_pip_to_usd(self, pip: float) -> float:
        """
        Convert pips to USD
        """
        return pip * 10
    

    def get_current_ask_price(self) -> float:
        """
        Get the current ask price
        """
        return game_manager.current_element['ask_close']

    def get_current_bid_price(self) -> float:
        """
        Get the current bid price
        """
        return game_manager.current_element['bid_close']
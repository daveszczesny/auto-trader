from typing import Tuple, Optional, Any

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from currency_converter import CurrencyConverter

from brooksai.env.models.trade import reset_open_trades, get_trade_profit, close_trade,\
    trigger_stop_or_take_profit, close_all_trades, check_margin, Trade, open_trades
from brooksai.env.models.constants import TradeType, ActionType, Punishment, Fee,\
    ApplicationConstants, Reward, action_type_mapping
from brooksai.env.models.action import Action, TradeAction
from brooksai.env.utils.converter import pip_to_profit

from brooksai.env.services.logs.logger import Logger

c = CurrencyConverter()

logger = Logger()

class SimpleForexEnv(gym.Env):
    """
    Custom starting Environment for forex trading
    This is a simplified version of the forex environment.
    The agent does not have to worry about stop loss, take profit, or lot size.
    The agent can only go long or short and close trades.
    The agent can only open one trade at a time.

    The market data is simplified, with noise reduction
    """


    def __init__(self, data: str, initial_balance: float = 1_000, render_mode: Optional[str] = None):

        self.data = pd.read_csv(data)

        # Environment variables
        self.n_steps = len(self.data)
        self.render_mode = render_mode

        # Observation space
        # balance, unrealised pnl, current price, current high, current low,
        # EMA 200, EMA 50, EMA 21, OPEN TRADES
        self.observation_space = spaces.Box(low=0,
                                            high=np.inf,
                                            shape=(9, ),
                                            dtype=np.float32)

        # Action space
        # action taken, lot size, stop loss, take profit, trade index to close
        self.action_space = spaces.Box(
            low = np.array([0.0, 0.01, -1.0, -1.0, 0.0], dtype=np.float32),
            high = np.array([1.0, 1.0, 1.0, 1.0, ApplicationConstants.SIMPLE_MAX_TRADES - 1], dtype=np.float32),
            dtype=np.float32
        )

        self.current_step: int = 0
        self.current_price: float = self.data.iloc[self.current_step]['bid_close']
        self.current_high: float = self.data.iloc[self.current_step]['bid_high']
        self.current_low: float = self.data.iloc[self.current_step]['bid_low']
        self.current_emas = self.data.iloc[self.current_step][['EMA_200', 'EMA_50', 'EMA_21']]

        self.reward: float = 0.0
        self.trade_window: int = ApplicationConstants.DEFAULT_TRADE_WINDOW
        self.done: bool = False

        # Agent variables
        self.initial_balance: float = initial_balance
        self.current_balance: float = initial_balance
        self.unrealised_pnl: float = 0.0
        self.previous_balance: float = 0.0

    def step(self, action: np.ndarray) -> Tuple[np.array, float, bool, bool, dict]:

        action: Action = self.construct_action(action)

        self.reward = 0.0

        self.current_price = self.data.iloc[self.current_step]['bid_close']
        self.current_high = self.data.iloc[self.current_step]['bid_high']
        self.current_low = self.data.iloc[self.current_step]['bid_low']
        self.current_emas = self.data.iloc[self.current_step][['EMA_200', 'EMA_50', 'EMA_21']]

        trigger_stop_or_take_profit(self.current_high, self.current_low)
        self.calculate_reward(action)
        self.previous_balance = self.current_balance

        self.apply_actions(action)

        self.unrealised_pnl = self._get_unrealized_pnl()

        self.done = self.current_step == self.n_steps - 1 or self.current_balance <= check_margin(0.01)

        if self.done:
            self.current_balance += close_all_trades(self.current_price)

        self.current_step += 1

        logger.log(f"Step: {self.current_step}, Balance: {self.current_balance}, Unrealised PnL: {self.unrealised_pnl}, Reward: {self.reward}, Trades Open: {len(open_trades)}")

        return self._get_observation(), self.reward, self.done, False, {}

    def render(self, mode: str = 'human') -> None:
        pass

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.array, dict]:
        """
        Reset the state of the environment to the inital state
        :param seed: int
        :param options: dict
        :return: Tuple[np.array, dict]
        """

        super().reset(seed=seed)

        # Reset environment variables
        self.current_step = 0
        self.current_price = self.data.iloc[self.current_step]['bid_close']
        self.current_high = self.data.iloc[self.current_step]['bid_high']
        self.current_low = self.data.iloc[self.current_step]['bid_low']
        self.current_emas = self.data.iloc[self.current_step][['EMA_200', 'EMA_50', 'EMA_21']]
        self.reward = 0.0
        self.trade_window = ApplicationConstants.DEFAULT_TRADE_WINDOW

        reset_open_trades()

        # Reset agent variables
        self.current_balance = self.initial_balance
        self.unrealised_pnl = 0.0

        logger.create_new_log_file()

        return self._get_observation(), {}

    def construct_action(self, raw_action: np.ndarray) -> Action:
        """
        Construct the Agent Action from raw action values
        :param raw_action: np.ndarray
        :return: Action
        """

        action_type = action_type_mapping.get(int(raw_action[0] * 3), ActionType.DO_NOTHING)

        if action_type in [ActionType.LONG, ActionType.SHORT]:
            if raw_action[1] is None:
                action = Action(action_type=ActionType.DO_NOTHING)
                self.trade_window -= 1
                return action

            lot_size: float = raw_action[1]
            stop_loss = float(raw_action[2]) * ApplicationConstants.TP_AND_SL_SCALE_FACTOR \
                if raw_action[2] > 0 else None
            take_profit = float(raw_action[3]) * ApplicationConstants.TP_AND_SL_SCALE_FACTOR \
                if raw_action[3] > 0 else None

            trade_action = TradeAction(
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=lot_size
            )

            action = Action(
                action_type=action_type,
                data=trade_action,
                trade=None
            )

            self.trade_window = ApplicationConstants.DEFAULT_TRADE_WINDOW
            return action

        elif action_type is ActionType.CLOSE:
            self.trade_window -= 1
            trade_index = int(raw_action[4])
            trade: Optional[Trade] = None
            if 0 <= trade_index < len(open_trades):
                trade = open_trades[trade_index]

            if trade is None:
                action = Action(action_type=ActionType.DO_NOTHING)
                return action
            
            action = Action(
                action_type=action_type,
                data=None,
                trade=trade
            )
            return action
        else:
            self.trade_window -= 1
            action = Action(action_type=action_type)
            return action

    def apply_actions(self, action: Action) -> None:
        """
        Apply Agent actions to the environment
        :param action: Action
        """

        if action.action_type is ActionType.DO_NOTHING:
            return

        if len(open_trades) < ApplicationConstants.SIMPLE_MAX_TRADES\
            and action.action_type in [ActionType.LONG, ActionType.SHORT]:

            Trade(
                lot_size=action.data.lot_size,
                open_price=self.current_price,
                trade_type=TradeType.LONG if action.action_type is ActionType.LONG else TradeType.SHORT,
                stop_loss=None,
                take_profit=None
            )

        elif action.action_type is ActionType.CLOSE:
            self.current_balance += close_trade(action.trade, self.current_price)


    def calculate_reward(self, action: Action) -> float:
        """
        Reward function for agent actions
        :param action: Action
        :return: float
        """

        # Check if agent tried to make a trade, but couldn't
        if action.action_type in [ActionType.LONG, ActionType.SHORT]:
            if len(open_trades) >= ApplicationConstants.SIMPLE_MAX_TRADES:
                self.reward -= Punishment.MAX_TRADES_REACHED
            else:
                self.reward += Reward.TRADE_OPENED

        elif action.action_type is ActionType.CLOSE:
            if action.trade is None:
                self.reward -= 1
            else:
                self.reward += Reward.TRADE_CLOSED

                if action.trade.ttl > 0:
                    self.reward += Reward.TRADE_CLOSED_WITHIN_TTL

                if get_trade_profit(action.trade, self.current_price) > 0:
                    self.reward += Reward.TRADE_CLOSED_IN_PROFIT

        if not self.done and self.current_step == self.n_steps - 1:
            self.reward += Reward.COMPLETED_RUN
        
        # Check if agent has run out of money to trade
        if self.current_balance < check_margin(0.01):
            self.reward -= Punishment.INSUFFICIENT_FUNDS

        # Check if Agent has a long lasting trade
        # if trade is open for more than permitted days, decrease the reward
        for trade in open_trades:
            trade.ttl -= 1
            if trade.ttl <= 0:
                self.reward -= Fee.EXTRA_DAY_FEE.value

        # Check margin call
        if self.current_balance <= (self._calc_sum_margin() - self.unrealised_pnl):
            self.current_balance += sum(close_trade(trade,self.current_balance) for trade in open_trades)
            self.reward -= Punishment.MARGIN_CALLED
        
        # Check if agent has not traded within the trade window
        if self.trade_window == 0:
            self.reward -= Punishment.NO_TRADE_WITHIN_WINDOW
        elif self.trade_window < 0:
            self.reward += Punishment.NO_TRADE_WITHIN_WINDOW * self.trade_window

    def apply_environment_rules(self) -> None:
        """
        Apply the environment rules
        """
        pass

    def _calc_sum_margin(self) -> float:
        return sum(trade.get_margin() for trade in open_trades)

    def _get_unrealized_pnl(self) -> float:
        return sum(
            pip_to_profit(self.current_price - trade.open_price, trade.lot_size) if
            trade.trade_type is TradeType.LONG else
            pip_to_profit(trade.open_price - self.current_price, trade.lot_size)
            for trade in open_trades
        )

    def _get_observation(self) -> np.array:
        """
        Get the observation of the environment
        """

        # balance, unrealised pnl, current price, current high, current low,
        # EMA 200, EMA 50, EMA 21, OPEN TRADES
        observation = np.array([
            self.current_balance,
            self.unrealised_pnl,
            self.current_price,
            self.current_high,
            self.current_low,
            *self.current_emas,
            len(open_trades)
        ], np.float32)

        return observation
from typing import Tuple, Optional, List

import pandas as pd

import torch
if torch.cuda.is_available():
    pass
else:
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

logger = Logger(mode='test')

# Agent improvment metrics
# This is not reset per epsiode, but for every run of training

agent_improvement_metric = {
    "win_rate": np.array([]),
    "average_win":  np.array([]),
    "average_loss":  np.array([]),
    "win_lose_ratio":  np.array([]),
    "steps": np.array([])
}


# pylint: disable=too-many-instance-attributes
class SimpleForexEnv(gym.Env):
    """
    Custom starting Environment for forex trading
    This is a simplified version of the forex environment.
    The agent does not have to worry about stop loss, take profit, or lot size.
    The agent can only go long or short and close trades.
    The agent can only open one trade at a time.

    The market data is simplified, with noise reduction
    """


    def __init__(self,
                 data: str,
                 initial_balance: float = 1_000,
                 render_mode: Optional[str] = None):

        self.data = pd.read_csv(data)
        self.data = torch.tensor(self.data.values, dtype=torch.float32)

        # Environment variables
        self.n_steps = len(self.data)
        self.render_mode = render_mode

        # Observation space
        # balance, unrealised pnl, current price, current high, current low,
        # EMA 200, EMA 50, EMA 21, OPEN TRADES
        self.observation_space = spaces.Box(low=0,
                                            high=np.inf,
                                            shape=(9, ),
                                            dtype=torch.float32)

        # Action space
        # action taken, lot size, stop loss, take profit, trade index to close
        self.action_space = spaces.Box(
            low = np.array([0.0, 0.01, -1.0, -1.0, 0.0], dtype=torch.float32),
            high = np.array([1.0, 1.0, 1.0, 1.0, ApplicationConstants.SIMPLE_MAX_TRADES - 1],
                            dtype=torch.float32),
            dtype=torch.float32
        )

        self.current_step: int = 0

        self._update_current_state()

            # self.current_price: float = self.data.iloc[self.current_step]['bid_close']
            # self.current_high: float = self.data.iloc[self.current_step]['bid_high']
            # self.current_low: float = self.data.iloc[self.current_step]['bid_low']
            # self.current_emas = self.data.iloc[self.current_step][['EMA_200', 'EMA_50', 'EMA_21']]

        self.previous_unrealized_pnl: List[float] = []
        self.reward: float = 0.0
        self.trade_window: int = ApplicationConstants.DEFAULT_TRADE_WINDOW
        self.done: bool = False
        self.max_reward: float = 5.0
        self.min_reward: float = -0.5

        # Agent variables
        self.initial_balance: float = initial_balance
        self.current_balance: float = initial_balance
        self.unrealised_pnl: float = 0.0
        self.previous_balance: float = 0.0


        # Action Tracker
        self.action_tracker = {
            "trades_opened": 0,
            "trades_closed": 0,
            "total_won": 0,
            "total_lost": 0,
            "times_won": 0,
            "times_lost": 0
        }

    def _update_current_state(self):
        if torch.cuda.is_available():
            self.data = self.data.cuda()

        self.current_price: float = self.data[self.current_step, 7].item()
        self.current_high: float = self.data[self.current_step, 6].item()
        self.current_low: float = self.data[self.current_step, 5].item()
        self.current_emas: Tuple[float, float, float] = (
            self.data[self.current_step, 11].item(),
            self.data[self.current_step, 12].item(),
            self.data[self.current_step, 13].item()
        )

    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, float, bool, bool, dict]:

        action: Action = self.construct_action(action)

        self.reward: float = 0.0

        self._update_current_state()

        trigger_stop_or_take_profit(self.current_high, self.current_low)
        self.calculate_reward(action)
        self.previous_balance = self.current_balance
        self.apply_actions(action)

        self.unrealised_pnl = self._get_unrealized_pnl()


        # Reset when
        # 1. Run is done
        # 2. Agent can't cover marin
        # 3. Losses over 1/4 of original balance
        self.done = self.current_step == self.n_steps - 2 or \
            self.current_balance * 0.5 <= check_margin(0.01) or \
                self.initial_balance * 0.75 >= self.current_balance - abs(self.unrealised_pnl if self.unrealised_pnl < 0 else 0) or \
                self.trade_window <= 0

        if self.done:
            self.current_balance += close_all_trades(self.current_price)
            self.previous_unrealized_pnl.clear()

            if self.trade_window <= 0:
                self.reward -= 1440

            # Log tracker
            
            average_win = float(self.action_tracker['total_won']) / float(self.action_tracker['trades_closed']) \
                if self.action_tracker['trades_closed'] > 0 else 0
            average_loss = float(self.action_tracker['total_lost']) / float(self.action_tracker['trades_closed']) \
                if self.action_tracker['trades_closed'] > 0 else 0

            self.reward += _calculate_agent_improvement(average_win, average_loss, self.action_tracker['times_won'], self.action_tracker['trades_closed'])

            logger.log_test('\nAction Tracker')
            logger.log_test(f'Trades opened: {self.action_tracker["trades_opened"]}')
            logger.log_test(f'Trades closed: {self.action_tracker["trades_closed"]}')
            logger.log_test(f'Average win: {round(average_win, 2)}')
            logger.log_test(f'Average loss: {round(average_loss, 2)}')
            win_rate = (self.action_tracker['times_won'] / self.action_tracker['trades_closed']) if self.action_tracker['trades_closed'] > 0 else 0
            logger.log_test(f'Win rate: {round(win_rate, 2)}')

        logger.log_test(f"{self.current_step}, {action.action_type.value}, {len(open_trades)}, "
                   f"{round(action.data.lot_size, 2) if action.data is not None else 0}, "
                   f"{round(self.current_price, 5)}, {round(self.current_low, 5)}, "
                   f"{round(self.current_high, 5)}, {round(self.current_balance, 2)}, "
                   f"{round(self.unrealised_pnl, 2)}, {round(self.reward, 2)}")
        

        logger.log_debug(f"Step: {self.current_step}, Action: {action.action_type}, "
                   f"Balance: {round(self.current_balance, 2)}, "
                   f"Unrealised PnL: {round(self.unrealised_pnl, 2)}, "
                   f"Reward: {round(self.reward, 3)}, Trades Open: {len(open_trades)}")
    
        self.current_step += 1

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

        agent_improvement_metric['steps'] = np.append(agent_improvement_metric['steps'], self.current_step)

        super().reset(seed=seed)

        # Reset environment variables
        self.current_step = 0
        self.current_price = self.data.iloc[self.current_step]['bid_close']
        self.current_high = self.data.iloc[self.current_step]['bid_high']
        self.current_low = self.data.iloc[self.current_step]['bid_low']
        self.current_emas = self.data.iloc[self.current_step][['EMA_200', 'EMA_50', 'EMA_21']]
        self.reward = 0.0
        self.trade_window = ApplicationConstants.DEFAULT_TRADE_WINDOW
        self.max_reward: float = 5.0
        self.min_reward: float = -0.5

        reset_open_trades()
        self.previous_unrealized_pnl.clear()

        # Reset agent variables
        self.current_balance = self.initial_balance
        self.unrealised_pnl = 0.0

        self.action_tracker = {
            "trades_opened": 0,
            "trades_closed": 0,
            "total_won": 0,
            "total_lost": 0,
            "times_won": 0,
            "times_lost": 0
        }

        logger.create_new_log_file()

        return self._get_observation(), {}

    def construct_action(self, raw_action: np.ndarray) -> Action:
        """
        Construct the Agent Action from raw action values
        :param raw_action: np.ndarray
        :return: Action
        """

        action_type = action_type_mapping.get(int(raw_action[0] * len(action_type_mapping)), \
                                              ActionType.DO_NOTHING)

        if action_type in [ActionType.LONG, ActionType.SHORT]:
            if raw_action[1] is None or len(open_trades) >= ApplicationConstants.SIMPLE_MAX_TRADES:
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
            trade_index = int(raw_action[4])
            trade: Optional[Trade] = None
            if 0 <= trade_index < len(open_trades):
                trade = open_trades[trade_index]

            if trade is None:
                action = Action(action_type=ActionType.DO_NOTHING)
                self.trade_window -= 1
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
                trade_type=TradeType.LONG if action.action_type is ActionType.LONG \
                    else TradeType.SHORT,
                stop_loss=None,
                take_profit=None
            )

            self.action_tracker['trades_opened'] += 1
            return

        elif action.action_type is ActionType.CLOSE:
            if action.trade is not None:
                logger.log_debug(f"Closing trade {action.trade.trade_type}."
                           f"Opened: {action.trade.open_price}, "
                           f"Closed: {self.current_price}. "
                           f"Profit: {get_trade_profit(action.trade, self.current_price)}")
                self.action_tracker['trades_closed'] += 1
                value = get_trade_profit(action.trade, self.current_price) - Fee.TRANSACTION_FEE
                if  value  > 0:
                    self.action_tracker['total_won'] += value
                    self.action_tracker['times_won'] += 1
                else:
                     self.action_tracker['total_lost'] += value
                     self.action_tracker['times_lost'] += 1
                self.current_balance += close_trade(action.trade, self.current_price)


    def calculate_reward(self, action: Action) -> float:
        """
        Reward function for agent actions
        :param action: Action
        :return: float
        """

        """
        Generally in trading, doing nothing is usually the right thing to do.
        So, we will give a small reward for doing nothing so that the agent
            learns to not over trade, and let trades run.

        We also don't want the agent to indefinitely hold trades. We want our agent
            to be an intraday trader. Taking small but consistent profits.
        So, we will give a small punishment for holding trades for too long.

        We will also punish the agent for invalid actions. The agent will only be permitted
            to have one trade open at a time. If the agent tries to open another trade
            while one is already open, the agent will be punished.
            or close a trade that does not exist, the agent will be punished.
        
        We will also punish the agent for closing trades too early. We want the agent to
            let trades run, and not close trades too early. This, however, depends on the trade,
            scalping is a valid trading strategy. But, we want the agent to learn to let trades run.
        So, we will punish the agent for closing trades too early, if the agent is not
            in profit.

        We will reward the agent for closing trades in profit
        To prevent the agent from holding trades too long we will punish the agent for holding trades
            past the trade ttl. The trade ttl is the number of minutes the agent can hold a trade.
        
        We will also punish the agent for holding trades that are not in profit, for too long. Trades
            can turn around, but we want the agent to learn to cut losses early. Again we want an
            intraday trading bot. Not an investor.

        We will also reward the agent for improving over time. We want the agent to learn from its
            mistakes and improve over time.

        """

        significant_loss_threshold = self.current_balance * 0.05
        significant_gain_threshold = 80

        # Check for invalid actions
        if len(open_trades) >= ApplicationConstants.SIMPLE_MAX_TRADES and \
                action.action_type in [ActionType.LONG, ActionType.SHORT]:
            self.reward -= Punishment.INVALID_ACTION

        elif len(open_trades) <= 0 and action.action_type is ActionType.CLOSE:
            self.reward -= Punishment.INVALID_ACTION

        elif len(open_trades) > 0 and action.action_type is ActionType.CLOSE:
            # Reward for closing trades within the trade ttl

            # Reward agent for closing a profitable scalp
            if open_trades[0].ttl >= ApplicationConstants.DEFAULT_TRADE_TTL - 5 and \
                get_trade_profit(open_trades[0], self.current_price) > Fee.TRANSACTION_FEE:
                self.reward += Reward.TRADE_CLOSED_IN_PROFIT

            # If scalp is not profitable, punish agent for closing too early
            elif open_trades[0].ttl >= ApplicationConstants.DEFAULT_TRADE_TTL - 5:
                self.reward -= Punishment.CLOSING_TOO_QUICK
            
            # Reward agent for closing trades within the trade ttl
            if 0 < open_trades[0].ttl <= ApplicationConstants.DEFAULT_TRADE_TTL - 5:
                self.reward += Reward.TRADE_CLOSED_WITHIN_TTL

                # Only rewarded if closed within ttl
                # Reward for closing trades in profit
                if get_trade_profit(open_trades[0], self.current_price) > Fee.TRANSACTION_FEE:
                    self.reward += Reward.TRADE_CLOSED_IN_PROFIT

        # Punish the agent for holding onto a lossing trade for too long (2 hours)
        if len(open_trades) > 0 and get_trade_profit(open_trades[0], self.current_price) < -(self.current_balance * 0.02) and \
            open_trades[0].ttl <= ApplicationConstants.DEFAULT_TRADE_TTL - 60:
            self.reward -= Punishment.HOLDING_LOSSING_TRADE
        
        # Punish the agent for holding a trade with a big loss
        if len(open_trades) > 0 and get_trade_profit(open_trades[0], self.current_price) < - significant_loss_threshold:
            self.reward -= Punishment.HOLDING_LOSSING_TRADE

        # Punish the agent for holding a trade with a big profit
        if len(open_trades) > 0 and get_trade_profit(open_trades[0], self.current_price) > significant_gain_threshold:
            self.reward -= Punishment.RISKY_HOLDING

        # Reward for doing nothing
        if action.action_type is ActionType.DO_NOTHING:
            if len(open_trades) > 0 and get_trade_profit(open_trades[0], self.current_price) < -significant_loss_threshold:
                pass
            else:
                self.reward += Reward.SMALL_REWARD_FOR_DOING_NOTHING

        if self.trade_window <= 0:
            self.reward -= Punishment.NO_TRADE_OPEN

        for trade in open_trades:
            # If trade is open for more than 10 days + 1 day overdraft, set reward to negative
            if trade.ttl <= ApplicationConstants.TRADE_TTL_OVERDRAFT_LIMIT:
                self.reward = - abs(2 * trade.ttl)
                return

        normalized_reward: float = 0
        if self.max_reward != self.min_reward:
            normalized_reward = (self.reward - self.min_reward) / (self.max_reward - self.min_reward) * 2 - 1
        else:
            normalized_reward = 0

        self.reward = normalized_reward

    def _calc_sum_margin(self) -> float:
        return sum(trade.get_margin() for trade in open_trades)

    def _get_unrealized_pnl(self) -> float:
        return sum(
            pip_to_profit(self.current_price - trade.open_price, trade.lot_size) if
            trade.trade_type is TradeType.LONG else
            pip_to_profit(trade.open_price - self.current_price, trade.lot_size)
            for trade in open_trades
        )

    def _get_observation(self) -> torch.Tensor:
        """
        Get the observation of the environment
        """
        # balance, unrealised pnl, current price, current high, current low,
        # EMA 200, EMA 50, EMA 21, OPEN TRADES
        observation = torch.tensor([
                self.current_balance,
                self.unrealised_pnl,
                self.current_price,
                self.current_high,
                self.current_low,
                *self.current_emas,
                len(open_trades)
                ], dtype=torch.float32)
        return observation.cuda() if torch.cuda.is_available() else observation



def _calculate_agent_improvement(average_win, average_loss, times_won, trades_closed) -> float:

    reward: float = 0.0

    max_array_size: int = 12_000

    if len(agent_improvement_metric['win_rate']) > max_array_size:
        agent_improvement_metric['win_rate'] = agent_improvement_metric['win_rate'][max_array_size/2:]

    if len(agent_improvement_metric['average_win']) > max_array_size:
        agent_improvement_metric['average_win'] = agent_improvement_metric['average_win'][max_array_size/2:]

    if len(agent_improvement_metric['average_loss']) > max_array_size:
        agent_improvement_metric['average_loss'] = agent_improvement_metric['average_loss'][max_array_size/2:]

    if len(agent_improvement_metric['win_lose_ratio']) > max_array_size:
        agent_improvement_metric['win_lose_ratio'] = agent_improvement_metric['win_lose_ratio'][max_array_size/2:]

    # Calculate win rate
    agent_improvement_metric['win_rate'] = np.append(agent_improvement_metric['win_rate'],
        round((times_won / trades_closed) \
                if trades_closed > 0 else 0, 2))

    # Calculate average win and loss
    agent_improvement_metric['average_win'] = np.append(agent_improvement_metric['average_win'], round(average_win, 2))
    agent_improvement_metric['average_loss'] = np.append(agent_improvement_metric['average_loss'], round(average_loss, 2))

    agent_improvement_metric['win_lose_ratio'] = np.append(agent_improvement_metric['win_lose_ratio'],
        (average_win / abs(average_loss)) if abs(average_loss) > 0 else 1
    )

    # Check if the ratio of average won and average loss is improving
    if len(agent_improvement_metric['win_lose_ratio']) > 1:
        if agent_improvement_metric['win_lose_ratio'][-1] > np.mean(agent_improvement_metric['win_lose_ratio'][:-1]):
             reward += Reward.AGENT_IMPROVED
        else:
            reward -= Punishment.AGENT_NOT_IMPROVING

    # If win rate is 0, the agent did not have a winning trade or didn't trade at all
    if agent_improvement_metric['win_rate'][-1] == 0:
        reward -= Punishment.NO_TRADE_OPEN

    # If the win rate was better than average
    if len(agent_improvement_metric['win_rate']) > 1:
        if agent_improvement_metric['win_rate'][-1] > np.mean(agent_improvement_metric['win_rate'][:-1]):
            reward += Reward.AGENT_IMPROVED
        else:
            reward -= Punishment.AGENT_NOT_IMPROVING

    # If average win was better than average
    if len(agent_improvement_metric['average_win']) > 1:
        if agent_improvement_metric['average_win'][-1] > np.mean(agent_improvement_metric['average_win'][:-1]):
            reward += Reward.BETTER_AVERAGE_TRADE
        else:
            reward -= Punishment.AGENT_NOT_IMPROVING

    # If average loss was smaller than best loss
    if len(agent_improvement_metric['average_loss']) > 1:
        if agent_improvement_metric['average_loss'][-1] < np.max(agent_improvement_metric['average_loss'][:-1]):
            reward += Reward.BETTER_AVERAGE_TRADE
        else:
            reward -= Punishment.AGENT_NOT_IMPROVING

    # Reward agent for longer run
    if len(agent_improvement_metric['steps']) > 1 and \
        agent_improvement_metric['steps'][-1] > np.max(agent_improvement_metric['steps'][:-1]):
        reward += Reward.AGENT_IMPROVED * 20
    elif len(agent_improvement_metric['steps']) == 1:
        pass
    else:
        reward -= Punishment.AGENT_NOT_IMPROVING * 5

    # If agent traded more than 1.5 trades average a day
    # Should prevent under and over trading
    average_trades_per_day = agent_improvement_metric['steps'][-1] / 1440
    if 1 < trades_closed / (average_trades_per_day if average_trades_per_day != 0 else 1) < 8:
        reward += Reward.AGENT_IMPROVED
    else:
        reward -= 1


    return reward
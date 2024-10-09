from typing import Tuple, Optional, List, Dict, Any

import dask.dataframe as dd

import torch
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
    "win_rate": torch.tensor([], dtype=torch.float32, device=ApplicationConstants.DEVICE),
    "average_win": torch.tensor([], dtype=torch.float32, device=ApplicationConstants.DEVICE),
    "average_loss": torch.tensor([], dtype=torch.float32, device=ApplicationConstants.DEVICE),
    "win_lose_ratio": torch.tensor([], dtype=torch.float32, device=ApplicationConstants.DEVICE),
    "steps": torch.tensor([], dtype=torch.float32, device=ApplicationConstants.DEVICE)
}


# pylint: disable=too-many-instance-attributes
class SimpleForexEnv(gym.Env):
    """
    Custom starting Environment for forex trading
    This is a simplified version of the forex environment.
    The agent does not have to worry about stop loss, take profit, or lot size.
    The agent can only go long or short and close trades.
    The agent can only open one trade at a time.
    """


    def __init__(self,
                 data: str,
                 initial_balance: float = 1_000.0,
                 render_mode: Optional[str] = None):

        self.data = dd.read_csv(data)
        self.data = self.data.select_dtypes(include=[float, int])
        self.data = self.data.to_dask_array(lengths=True)
        self.data = self.data.compute()
        self.data = torch.tensor(self.data, dtype=torch.float32, device=ApplicationConstants.DEVICE)

        # Environment variables
        self.n_steps = len(self.data)
        self.render_mode = render_mode

        # Observation space
        # balance, unrealised pnl, current price, current high, current low,
        # EMA 200, EMA 50, EMA 21, OPEN TRADES
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(9, ),
            dtype=np.float32
        )

        # Action space
        # action taken, lot size, stop loss, take profit
        self.action_space = spaces.Box(
            low = np.array([0.0, 0.01, -1.0, -1.0], dtype=np.float32),
            high = np.array([1.0, 1.0, 1.0, 1.0],
                            dtype=np.float32),
            dtype=np.float32
        )

        self.current_step: int = 0

        self._update_current_state()

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
        if torch.cuda.is_available() and not self.data.is_cuda:
            self.data = self.data.cuda()

        self.current_price: float = float(self.data[self.current_step, 6].item())
        self.current_high: float = float(self.data[self.current_step, 5].item())
        self.current_low: float = float(self.data[self.current_step, 4].item())
        self.current_emas: Tuple[float, float, float] = (
            float(self.data[self.current_step, 10].item()),
            float(self.data[self.current_step, 11].item()),
            float(self.data[self.current_step, 12].item())
        )

    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, float, bool, bool, dict]:

        action: Action = self.construct_action(action)

        self.reward: float = 0.0

        self._update_current_state()

        trigger_stop_or_take_profit(self.current_high, self.current_low)
        self.calculate_reward(action)
        self.previous_balance = self.current_balance
        self.apply_actions(action)

        self.unrealised_pnl = float(self._get_unrealized_pnl())


        # Reset when
        # 1. Run is done
        # 2. Agent can't cover marin
        # 3. Losses over 1/4 of original balance
        self.done = self.current_step == self.n_steps - 2 or \
            self.current_balance * 0.5 <= check_margin(0.01) or \
                self.initial_balance * 0.75 >= self.current_balance - \
                    abs(self.unrealised_pnl if self.unrealised_pnl < 0 else 0) or \
                self.trade_window <= 0

        if self.done:
            self.current_balance += close_all_trades(self.current_price)
            self.previous_unrealized_pnl.clear()

            if self.trade_window <= 0:
                self.reward -= 1440

            # Log tracker
            average_win = float(
                self.action_tracker['total_won']) / float(self.action_tracker['trades_closed']
                ) if self.action_tracker['trades_closed'] > 0 else 0
            average_loss = float(
                self.action_tracker['total_lost']) / float(self.action_tracker['trades_closed']
                ) if self.action_tracker['trades_closed'] > 0 else 0

            self.reward += _calculate_agent_improvement(
                average_win,
                average_loss,
                self.action_tracker['times_won'],
                self.action_tracker['trades_closed']
            )

            #Logging would use variables from the GPU, how can we log this?
            logger.log_test('\nAction Tracker')
            logger.log_test(f'Trades opened: {self.action_tracker["trades_opened"]}')
            logger.log_test(f'Trades closed: {self.action_tracker["trades_closed"]}')
            logger.log_test(f'Average win: {average_win}')
            logger.log_test(f'Average loss: {average_loss}')
            win_rate = (self.action_tracker['times_won'] / self.action_tracker['trades_closed']) \
                if self.action_tracker['trades_closed'] > 0 else 0
            logger.log_test(f'Win rate: {win_rate}')

        logger.log_test(f"{self.current_step}, "
                        f"{action.action_type.value}, "
                        f"{len(open_trades)}, "
                        f"{round(action.data.lot_size, 2) if action.data is not None else 0}, "
                        f"{round(self.current_price, 5)}, "
                        f"{round(self.current_low, 5)}, "
                        f"{round(self.current_high, 5)}, "
                        f"{round(self.current_balance, 2)}, "
                        f"{round(self.unrealised_pnl, 2)}, "
                        f"{self.reward}"
        )

        logger.log_debug(f"Step: {self.current_step}, Action: {action.action_type}, "
                        f"Balance: {self.current_balance}, "
                        f"Unrealised PnL: {self.unrealised_pnl}, "
                        f"Reward: {self.reward}, "
                        f"Trades Open: {len(open_trades)}")

        self.current_step += 1

        return self._get_observation(), self.reward, self.done, False, {}

    def render(self, mode: str = 'human') -> None:
        pass

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the state of the environment to the inital state
        :param seed: int
        :param options: dict
        :return: Tuple[np.array, dict]
        """

        agent_improvement_metric['steps'] = torch.cat(
            (agent_improvement_metric['steps'],
            torch.tensor(
                [self.current_step],
                dtype=torch.float32,
                device=ApplicationConstants.DEVICE)
            )
        )

        super().reset(seed=seed)

        # Reset environment variables
        self.current_step = 0
        self._update_current_state()
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

    def construct_action(self, raw_action: torch.Tensor) -> Action:
        """
        Construct the Agent Action from raw action values
        :param raw_action: torch.Tensor
        :return: Action
        """

        def get_action_type(raw_action: torch.Tensor) -> ActionType:
            index = int(raw_action[0].item() * len(action_type_mapping))
            return action_type_mapping.get(index, ActionType.DO_NOTHING)

        def is_invalid_action(action_type: ActionType, raw_action: torch.Tensor) -> bool:
            return (
                action_type in [ActionType.LONG, ActionType.SHORT] and \
                    (raw_action[1].item() <= 0 or \
                        len(open_trades) >= ApplicationConstants.SIMPLE_MAX_TRADES)) or \
                action_type is ActionType.CLOSE and len(open_trades) <= 0

        def create_trade_action(raw_action: torch.Tensor) -> TradeAction:
            lot_size = raw_action[1].item()
            stop_loss = (raw_action[2].item() * ApplicationConstants.TP_AND_SL_SCALE_FACTOR
                if raw_action[2] > 0 else None)
            take_profit = (raw_action[3].item() * ApplicationConstants.TP_AND_SL_SCALE_FACTOR
                if raw_action[3].item() > 0 else None)

            return TradeAction(
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=lot_size
            )

        action_type = get_action_type(raw_action)

        if is_invalid_action(action_type, raw_action):
            self.trade_window -= 1
            return Action(action_type=ActionType.DO_NOTHING)

        if action_type in [ActionType.LONG, ActionType.SHORT]:
            trade_action = create_trade_action(raw_action)
            return Action(action_type=action_type, data=trade_action, trade=None)

        if action_type is ActionType.CLOSE:
            return Action(action_type=action_type, data=None, trade=open_trades[0])

        return Action(action_type=ActionType.DO_NOTHING)

    def apply_actions(self, action: Action) -> None:
        """
        Apply Agent actions to the environment
        :param action: Action
        """

        if action.action_type in [ActionType.LONG, ActionType.SHORT]:
            Trade(
                lot_size=action.data.lot_size,
                open_price=self.current_price,
                trade_type=TradeType.LONG if action.action_type is ActionType.LONG \
                    else TradeType.SHORT,
                stop_loss=None,
                take_profit=None
            )
            self.action_tracker['trades_opened'] += 1

        else:
            if not action.trade:
                return

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
        # TODO Refactor this function

        significant_loss_threshold = self.current_balance * 0.05
        significant_gain_threshold = 80
        trade_profit = get_trade_profit(open_trades[0], self.current_price) if open_trades else 0

        is_trade_open = len(open_trades) > 0
        invalid_trade = is_trade_open and action.action_type in [ActionType.LONG, ActionType.SHORT]
        invalid_close = not is_trade_open and action.action_type is ActionType.CLOSE

        # Check invalid actions
        if invalid_trade or invalid_close:
            self.reward -= Punishment.INVALID_ACTION

        if is_trade_open:
            if action.action_type is ActionType.CLOSE:
                
                self.reward += Reward.TRADE_CLOSED

                ttl = open_trades[0].ttl
                if ttl >= ApplicationConstants.DEFAULT_TRADE_TTL - 5:
                    # Reward agent for closing a profitable scalp
                    if trade_profit > Fee.TRANSACTION_FEE:
                        self.reward += Reward.TRADE_CLOSED_IN_PROFIT
                    # If scalp is not profitable, punish agent for closing too early
                    else:
                        self.reward -= Punishment.CLOSING_TOO_QUICK
                if 0 < ttl <= ApplicationConstants.DEFAULT_TRADE_TTL - 5:
                    self.reward += Reward.TRADE_CLOSED_WITHIN_TTL

                    if trade_profit > Fee.TRANSACTION_FEE:
                        self.reward += Reward.TRADE_CLOSED_IN_PROFIT

                if ttl < 0:
                    self.reward -= Punishment.HOLDING_TRADE_TOO_LONG

                if ttl < -ApplicationConstants.TRADE_TTL_OVERDRAFT_LIMIT:
                    self.reward -= Punishment.TRADE_HELD_TOO_LONG * 5

            # Punish the agent for holding onto a lossing trade for too long (1 hours)
            if trade_profit < -(self.current_balance * 0.02) and \
                open_trades[0].ttl <= ApplicationConstants.DEFAULT_TRADE_TTL - 60:
                self.reward -= Punishment.HOLDING_LOSSING_TRADE
            # Punish the agent for holding a trade with a big loss
            if trade_profit < -significant_loss_threshold:
                self.reward -= Punishment.HOLDING_LOSSING_TRADE
            # Reward for doing nothing
            elif action.action_type is ActionType.DO_NOTHING:
                self.reward += Reward.SMALL_REWARD_FOR_DOING_NOTHING
            # Punish the agent for holding a trade with a big profit
            if trade_profit > significant_gain_threshold:
                self.reward -= Punishment.RISKY_HOLDING

        if self.trade_window <= 0:
            self.reward -= Punishment.NO_TRADE_OPEN


    def _calc_sum_margin(self) -> float:
        return sum(trade.get_margin() for trade in open_trades)

    def _get_unrealized_pnl(self) -> float:
        return float(sum(
            pip_to_profit(self.current_price - trade.open_price, trade.lot_size) if
            trade.trade_type is TradeType.LONG else
            pip_to_profit(trade.open_price - self.current_price, trade.lot_size)
            for trade in open_trades
        ))

    def _get_observation(self) -> np.ndarray:
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
                ], dtype=torch.float32, device=ApplicationConstants.DEVICE)
        return observation.cpu().numpy()


def _calculate_agent_improvement(average_win, average_loss, times_won, trades_closed) -> float:
    reward: float = 0.0
    max_array_size: int = 12_000

    # Truncate arrays if they exceed max array length
    for key in ['win_rate', 'average_win', 'average_loss', 'win_lose_ratio']:
        if agent_improvement_metric[key].size(0) > max_array_size:
            agent_improvement_metric[key] = agent_improvement_metric[key][max_array_size // 2:]


    # Calculate win rate
    win_rate = (times_won / trades_closed) if trades_closed > 0 else 0
    agent_improvement_metric['win_rate'] = torch.cat((
        agent_improvement_metric['win_rate'],
        torch.tensor(
            [win_rate],
            dtype=torch.float32,
            device=ApplicationConstants.DEVICE
        )
    ))

    # Calculate average win and loss
    agent_improvement_metric['average_win'] = torch.cat((
        agent_improvement_metric['average_win'],
        torch.tensor(
            [average_win],
            dtype=torch.float32,
            device=ApplicationConstants.DEVICE
        )
    ))

    agent_improvement_metric['average_loss'] = torch.cat((
        agent_improvement_metric['average_loss'],
        torch.tensor(
            [average_loss],
            dtype=torch.float32,
            device=ApplicationConstants.DEVICE
        )
    ))

    # Calculate average win and loss
    win_lose_ratio = (average_win / abs(average_loss)) if abs(average_loss) > 0 else 1
    agent_improvement_metric['win_lose_ratio'] = torch.cat((
        agent_improvement_metric['win_lose_ratio'],
        torch.tensor(
            [win_lose_ratio],
            dtype=torch.float32,
            device=ApplicationConstants.DEVICE
        )
    ))


    # Vectorized reward calculation
    win_lose_ratio_improved = (
        agent_improvement_metric['win_lose_ratio'][-1] > \
            agent_improvement_metric['win_lose_ratio'][:-1].mean()
    ).float()

    win_rate_improved = (
        agent_improvement_metric['win_rate'][-1] > \
            agent_improvement_metric['win_rate'][:-1].mean()
    ).float()

    average_win_improved = (
        agent_improvement_metric['average_win'][-1] > \
            agent_improvement_metric['average_win'][:-1].mean()
    ).float()

    reward += Reward.AGENT_IMPROVED * win_lose_ratio_improved
    reward -= Punishment.AGENT_NOT_IMPROVING * (1 - win_lose_ratio_improved)
    reward -= Punishment.NO_TRADE_OPEN * (agent_improvement_metric['win_rate'][-1] == 0).float()
    reward += Reward.AGENT_IMPROVED * win_rate_improved
    reward -= Punishment.AGENT_NOT_IMPROVING * (1 - win_rate_improved)
    reward += Reward.AGENT_IMPROVED * average_win_improved
    reward -= Punishment.AGENT_NOT_IMPROVING * (1 - average_win_improved)

    return reward

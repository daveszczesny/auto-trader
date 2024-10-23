from typing import Tuple, Optional, List, Dict, Any

import dask.dataframe as dd

import torch
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from currency_converter import CurrencyConverter

from brooksai.models.trade import reset_open_trades,\
    trigger_stop_or_take_profit, close_all_trades, open_trades
from brooksai.models.constants import TradeType, ApplicationConstants
from brooksai.models.action import Action
from brooksai.utils.converter import pip_to_profit
from brooksai.utils.action import ActionBuilder, ActionApply
from brooksai.utils.reward import RewardFunction

from brooksai.services.logs.logger import Logger

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
                 initial_balance: float = ApplicationConstants.INITIAL_BALANCE,
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
            low = np.array([0.0, 0, 0, 0], dtype=np.float32),
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
        # Construct the action from agent input
        action: Action = ActionBuilder.construct_action(action)
        trigger_stop_or_take_profit(self.current_high, self.current_low)
        value, tw = ActionApply.apply_action(action,
                                             current_price=self.current_price,
                                             trade_window=self.trade_window)

        self.current_balance += value
        self.trade_window = tw

        self.reward = RewardFunction.calculate_reward(
            action,
            self.current_price
        )

        self.unrealised_pnl = float(self._get_unrealized_pnl())

        self._update_current_state()

        self._is_done()

        # Log the step
        logger.log_test(f"{self.current_step}, "
                        f"{action.action_type.value}, "
                        f"{len(open_trades)}, "
                        f"{round(action.trade_data.lot_size, 2) if action.trade_data is not None else 0}, "
                        f"{round(self.current_price, 5)}, "
                        f"{round(self.current_low, 5)}, "
                        f"{round(self.current_high, 5)}, "
                        f"{round(self.current_balance, 2)}, "
                        f"{round(self.unrealised_pnl, 2)}, "
                        f"{self.reward}"
        )

        self.current_step += 1

        return self._get_observation(), self.reward, self.done, False, {}

    def render(self, mode: str = 'human') -> None:
        pass

    def reset(self,
              *,
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

        ActionApply.reset_tracker()

        logger.create_new_log_file()
        return self._get_observation(), {}


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


    def _is_done(self):
        self.done = bool(
            self.initial_balance * 0.75 >= self.current_balance -
            abs(self.unrealised_pnl if self.unrealised_pnl < 0 else 0)
            or self.trade_window < 0
        )

        if self.done:
            self.current_balance += close_all_trades(self.current_price)
            self.previous_unrealized_pnl.clear()


            if self.trade_window < 0:
                self.reward -= 0.2

            average_win = float(
                ActionApply.get_action_tracker('total_won')) / float(ActionApply.get_action_tracker('trades_closed')
                ) if ActionApply.get_action_tracker('trades_closed') > 0 else 0
            average_loss = float(
                ActionApply.get_action_tracker('total_lost')) / float(ActionApply.get_action_tracker('trades_closed')
                ) if ActionApply.get_action_tracker('trades_closed') > 0 else 0

            self.reward += RewardFunction.calculate_agent_improvement(
                average_win,
                average_loss,
                ActionApply.get_action_tracker('times_won'),
                ActionApply.get_action_tracker('trades_closed'),
                self.current_step
            )

            # Log tracker
            logger.log_test('\nAction Tracker')
            logger.log_test(f'Trades opened: {ActionApply.get_action_tracker("trades_opened")}')
            logger.log_test(f'Trades closed: {ActionApply.get_action_tracker("trades_closed")}')
            logger.log_test(f'Average win: {average_win}')
            logger.log_test(f'Average loss: {average_loss}')
            win_rate = (ActionApply.get_action_tracker('times_won') / ActionApply.get_action_tracker('trades_closed')) \
                if ActionApply.get_action_tracker('trades_closed') > 0 else 0
            logger.log_test(f'Win rate: {win_rate}')

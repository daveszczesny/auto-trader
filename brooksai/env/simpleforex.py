"""
@article{towers2024gymnasium,
  title={Gymnasium: A Standard Interface for Reinforcement Learning Environments},
  author={Towers, Mark and Kwiatkowski, Ariel and Terry, Jordan and Balis, John U and De Cola, Gianluca and Deleu, Tristan and Goul{\~a}o, Manuel and Kallinteris, Andreas and Krimmel, Markus and KG, Arjun and others},
  journal={arXiv preprint arXiv:2407.17032},
  year={2024}
}
"""


from typing import Tuple, Optional, Dict, Any

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
                 data: str | dd.DataFrame,
                 initial_balance: float = ApplicationConstants.INITIAL_BALANCE,
                 render_mode: Optional[str] = None,
                 split: bool = False) -> None:

        if split:
            self.data = data
        else:
            self.data = dd.read_csv(data)
            self.data = self.data.select_dtypes(include=[float, int])
            self.data = self.data.to_dask_array(lengths=True)
            self.data = self.data.compute()
            self.data = torch.tensor(self.data, dtype=torch.float32, device=ApplicationConstants.DEVICE)

        # Environment variables
        self.n_steps = len(self.data) # number of steps in the dataset
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

        # Possible starting point is 80% of the total steps.
        # The agent will start at a random point within this range
        self._possible_starting_steps: int = int(self.n_steps * 0.8)
        self.current_step: int = 0

        self._update_current_state()

        self.reward: float = 0.0
        self.trade_window: int = ApplicationConstants.DEFAULT_TRADE_WINDOW
        self.done: bool = False

        # Agent variables
        self.initial_balance: float = initial_balance
        self.current_balance: float = initial_balance
        self.unrealised_pnl: float = 0.0


    def _update_current_state(self):
        """
        Update the current state of the environment
            including current price, high, low, and EMAs
        """

        # Data mapping
        HIGH_PRICE = 0 # bid_high
        LOW_PRICE = 1 # bid_low
        CLOSE_PRICE = 2 # bid_close
        EMA_21_PRICE = 3 # EMA 21
        EMA_50_PRICE = 4 # EMA 50
        EMA_200_PRICE = 5 # EMA 200

        if torch.cuda.is_available() and not self.data.is_cuda:
            self.data = self.data.cuda()

        self.current_price: float = float(self.data[self.current_step, CLOSE_PRICE].item())
        self.current_high: float = float(self.data[self.current_step, HIGH_PRICE].item())
        self.current_low: float = float(self.data[self.current_step, LOW_PRICE].item())
        self.current_emas: Tuple[float, float, float] = (
            float(self.data[self.current_step, EMA_21_PRICE].item()),
            float(self.data[self.current_step, EMA_50_PRICE].item()),
            float(self.data[self.current_step, EMA_200_PRICE].item())
        )

    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, float, bool, bool, dict]:
        """
        Method to process a step in the environment.
        :param action: np.ndarray
        :return: Tuple[torch.Tensor, float, bool, bool, dict], observation, reward, done, _, info
        """
        # Construct the action from agent input
        action: Action = ActionBuilder.construct_action(action)

        # Not needed right now, but will eventually once sl & tp are used by agent
        trigger_stop_or_take_profit(self.current_high, self.current_low)

        # Calculate the reward for step
        self.reward = self._calculate_reward(action)

        # Apply the action to the environment
        value, tw = ActionApply.apply_action(action,
                                             current_price=self.current_price,
                                             trade_window=self.trade_window)

        self.current_balance += value
        self.trade_window = tw


        self.unrealised_pnl = float(self._get_unrealized_pnl())

        self._update_current_state()

        # Check if the episode is done, if so, process the episode
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
        """
        Not rendering anything directly in the environment
        """
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

        # Close all trades
        reset_open_trades()

        # Reset agent variables
        self.current_balance = self.initial_balance
        self.unrealised_pnl = 0.0

        ActionApply.reset_tracker()
        RewardFunction.reset_rewards()

        logger.create_new_log_file()
        return self._get_observation(), {}

    def _get_unrealized_pnl(self) -> float:
        """
        Calculate the unrealized profit or loss of the open trades
        """

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

    def _calculate_reward(self, action: Action) -> float:
        if agent_improvement_metric['steps'].numel() > 0:
            best_step = agent_improvement_metric['steps']
            best_step = best_step.to(torch.int32)
            best_step = torch.max(best_step).item()
        else:
            best_step = 0

        return RewardFunction.get_reward(action,
                                         self.current_price,
                                         self.current_step,
                                         best_step)


    def _is_done(self):
        """
        Check if the episode is done
        """

        """
        Conditions for episode to be done:
        1. 75% of initial balance is lost
        2. Trade window is negative
        """
        self.done = bool(
            self.initial_balance * 0.5 >= self.current_balance -
            abs(self.unrealised_pnl if self.unrealised_pnl < 0 else 0) or
            self.current_step >= self.n_steps - 1
        )

        # If the episode is done, close all trades
        #   calculate the final balance
        #   and calculate the reward
        if self.done:
            self.current_balance += close_all_trades(self.current_price)

            if self.trade_window < 0 and ActionApply.get_action_tracker('trades_opened') <= 0:
                self.reward -= 1000
            
            if self.current_balance > self.initial_balance:
                self.reward += 100

            average_win = float(
                ActionApply.get_action_tracker('total_won')) / float(ActionApply.get_action_tracker('trades_closed')
                ) if ActionApply.get_action_tracker('trades_closed') > 0 else 0
            average_loss = float(
                ActionApply.get_action_tracker('total_lost')) / float(ActionApply.get_action_tracker('trades_closed')
                ) if ActionApply.get_action_tracker('trades_closed') > 0 else 0


            # Log tracker
            logger.log_test('\nAction Tracker')
            logger.log_test(f'Trades opened: {ActionApply.get_action_tracker("trades_opened")}')
            logger.log_test(f'Trades closed: {ActionApply.get_action_tracker("trades_closed")}')
            logger.log_test(f'Average win: {average_win}')
            logger.log_test(f'Average loss: {average_loss}')
            win_rate = (ActionApply.get_action_tracker('times_won') / ActionApply.get_action_tracker('trades_closed')) \
                if ActionApply.get_action_tracker('trades_closed') > 0 else 0
            logger.log_test(f'Win rate: {win_rate}')

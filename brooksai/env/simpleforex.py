from typing import Tuple, Optional, List

import dask.dataframe as dd

import torch
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from currency_converter import CurrencyConverter

from brooksai.env.models.gputrade import GPUTrade, check_margin

from brooksai.env.models.constants import ActionType,\
    ApplicationConstants, action_type_mapping, \
    GPUPunishment, GPUReward, GPUConstants
from brooksai.env.models.action import Action, TradeAction

from brooksai.env.services.logs.logger import Logger

c = CurrencyConverter()

logger = Logger(mode='test')

# Agent improvment metrics
# This is not reset per epsiode, but for every run of training

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Stored on the GPU
agent_improvement_metric = {
    "win_rate": torch.tensor([], dtype=torch.float32, device=DEVICE),
    "average_win": torch.tensor([], dtype=torch.float32, device=DEVICE),
    "average_loss": torch.tensor([], dtype=torch.float32, device=DEVICE),
    "win_lose_ratio": torch.tensor([], dtype=torch.float32, device=DEVICE),
    "steps": torch.tensor([], dtype=torch.float32, device=DEVICE)
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

        self.device = DEVICE

        self.data = dd.read_csv(data)
        self.data = self.data.select_dtypes(include=[float, int])
        self.data = self.data.to_dask_array(lengths=True)
        self.data = self.data.compute()
        self.data = torch.tensor(self.data, dtype=torch.float32, device=self.device)  # moves to GPU

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
        self.current_step_gpu = torch.tensor(self.current_step, dtype=torch.float32, device=self.device)

        self._update_current_state()

        self.previous_unrealized_pnl: torch.Tensor = torch.tensor([], dtype=torch.float32, device=self.device)
        self.reward: torch.Tensor = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.trade_window: int = ApplicationConstants.DEFAULT_TRADE_WINDOW
        self.done: bool = False

        # Agent variables
        self.initial_balance: float = initial_balance
        self.current_balance: float = initial_balance
        self.unrealised_pnl: float = 0.0
        self.previous_balance: float = 0.0


        self.trade = GPUTrade()


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
        #TODO Move to GPU
        step = None
        if torch.cuda.is_available() and not self.data.is_cuda:
            self.data = self.data.cuda()
            # use gpu step
        
        if torch.cuda.is_available():
            step = torch.tensor(self.current_step_gpu, dtype=torch.int64, device=self.device)
        else:
            step = int(self.current_step)
    
        self.current_price: float = float(self.data[step, 6].item())
        self.current_high: float = float(self.data[step, 5].item())
        self.current_low: float = float(self.data[step, 4].item())
        self.current_emas: Tuple[float, float, float] = (
            float(self.data[step, 10].item()),
            float(self.data[step, 11].item()),
            float(self.data[step, 12].item())
        )

    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, float, bool, bool, dict]:

        action: Action = self.construct_action(action)

        self.reward: float = 0.0

        self._update_current_state()

        self.trade.trigger_sl_or_tp(self.current_high, self.current_low)
        self.calculate_reward(action)
        self.previous_balance = self.current_balance
        self.apply_actions(action)

        self.unrealised_pnl = float(self._get_unrealized_pnl())


        # Reset when
        # 1. Run is done
        # 2. Agent can't cover marin
        # 3. Losses over 1/4 of original balance
        self.done = self.current_step == self.n_steps - 2 or \
            self.current_balance * 0.5 <= check_margin(0.01)  or \
                self.initial_balance * 0.75 >= self.current_balance - abs(self.unrealised_pnl if self.unrealised_pnl < 0 else 0) or \
                self.trade_window <= 0

        if self.done:
            self.current_balance += self.trade.close_trade(self.current_price).item()
            self.previous_unrealized_pnl = torch.tensor([], dtype=torch.float32, device=self.device)

            if self.trade_window <= 0:
                self.reward -= 1440

            # Log tracker
            
            average_win = float(self.action_tracker['total_won']) / float(self.action_tracker['trades_closed']) \
                if self.action_tracker['trades_closed'] > 0 else 0
            average_loss = float(self.action_tracker['total_lost']) / float(self.action_tracker['trades_closed']) \
                if self.action_tracker['trades_closed'] > 0 else 0

            self.reward += _calculate_agent_improvement(average_win, average_loss, self.action_tracker['times_won'], self.action_tracker['trades_closed'])

            #Logging would use variables from the GPU, how can we log this?
        #     logger.log_test('\nAction Tracker')
        #     logger.log_test(f'Trades opened: {self.action_tracker["trades_opened"]}')
        #     logger.log_test(f'Trades closed: {self.action_tracker["trades_closed"]}')
        #     logger.log_test(f'Average win: {average_win}')
        #     logger.log_test(f'Average loss: {average_loss}')
        #     win_rate = (self.action_tracker['times_won'] / self.action_tracker['trades_closed']) \
        #         if self.action_tracker['trades_closed'] > 0 else 0
        #     logger.log_test(f'Win rate: {win_rate}')

        # logger.log_test(f"{self.current_step}, {action.action_type.value}, "
        #                 f"{action.data.lot_size if action.data is not None else 0}, "
        #                 f"{self.current_price}, "
        #                 f"{self.current_low}, "
        #                 f"{self.current_high}, "
        #                 f"{self.current_balance}, "
        #                 f"{self.unrealised_pnl}, "
        #                 f"{self.reward}"
        # )

        # logger.log_debug(f"Step: {self.current_step}, Action: {action.action_type}, "
        #                 f"Balance: {self.current_balance}, "
        #                 f"Unrealised PnL: {self.unrealised_pnl}, "
        #                 f"Reward: {self.reward}, "
        # )

        self.current_step += 1
        self.current_step_gpu += 1

        return self._get_observation(), self.reward, self.done, False, {}

    def render(self, mode: str = 'human') -> None:
        pass

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the state of the environment to the inital state
        :param seed: int
        :param options: dict
        :return: Tuple[np.array, dict]
        """

        agent_improvement_metric['steps'] = torch.cat((agent_improvement_metric['steps'], torch.tensor([self.current_step], dtype=torch.float32, device=self.device)))

        super().reset(seed=seed)

        # Reset environment variables
        self.current_step = 0
        self._update_current_state()
        self.reward = 0.0
        self.trade_window = ApplicationConstants.DEFAULT_TRADE_WINDOW

        self.trade.close_trade(0.0)
        self.previous_unrealized_pnl = torch.tensor([], dtype=torch.float32, device=self.device)

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
        :param raw_action: np.ndarray
        :return: Action
        """

        action_type = action_type_mapping.get(int(raw_action[0].item() * len(action_type_mapping)), \
                                              ActionType.DO_NOTHING)

        if action_type in [ActionType.LONG, ActionType.SHORT]:

            if raw_action[1].item() <= 0 or self.trade.is_open:
                action = Action(action_type=ActionType.DO_NOTHING)
                self.trade_window -= 1
                return action

            lot_size: float = raw_action[1].item()
            stop_loss = float(raw_action[2].item()) * ApplicationConstants.TP_AND_SL_SCALE_FACTOR \
                if raw_action[2] > 0 else None
            take_profit = float(raw_action[3].item()) * ApplicationConstants.TP_AND_SL_SCALE_FACTOR \
                if raw_action[3].item() > 0 else None

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
            if self.trade.is_open:

                action = Action(
                    action_type=action_type,
                    data=None,
                    trade=self.trade
                )
                return action
            else:
                action = Action(action_type=ActionType.DO_NOTHING)
                self.trade_window -= 1
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

        if not self.trade.is_open \
            and action.action_type in [ActionType.LONG, ActionType.SHORT]:

            self.trade.open_trade(
                lot_size=torch.tensor(action.data.lot_size, dtype=torch.float32, device=self.device),
                open_price=torch.tensor(self.current_price, dtype=torch.float32, device=self.device),
                trade_type=torch.tensor(1 if action.action_type is ActionType.LONG else 0, dtype=torch.float32, device=self.device),
                stop_loss=None,
                take_profit=None
            )

            self.action_tracker['trades_opened'] += 1
            return

        elif action.action_type is ActionType.CLOSE:
            if action.trade is not None:
                # logger.log_debug(f"Closing trade {action.trade.trade_type}."
                #            f"Opened: {action.trade._open_price}, "
                #            f"Closed: {self.current_price}. "
                #            f"Profit: {get_trade_profit(action.trade, self.current_price)}")
                self.action_tracker['trades_closed'] += 1
                value = self.trade.close_trade(self.current_price).item()  # Convert tensor to float
                if value > 0:
                    self.action_tracker['total_won'] += value
                    self.action_tracker['times_won'] += 1
                else:
                    self.action_tracker['total_lost'] += value
                    self.action_tracker['times_lost'] += 1

                self.current_balance += value


    """
    Can we vectorize the reward function? (yes)
    If we can we should put it on the GPU. (maybe)
    """

    def calculate_reward(self, action: Action) -> float:
        """
        Reward function for agent actions
        :param action: Action
        :return: float
        """

        significant_loss_threshold = self.initial_balance * 0.05
        significant_gain_theshold = 80

        # initialize reward tensor
        reward_tensor = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        trade_profits = self.trade.calculate_pnl(self.current_price)
        ttls = self.trade.ttl

        # Check for invalid actions
        invalid_long_short = self.trade.is_open and action.action_type in [ActionType.LONG, ActionType.SHORT]
        invalid_close = not self.trade.is_open and action.action_type is ActionType.CLOSE

        if invalid_long_short or invalid_close:
            reward_tensor -= GPUPunishment.INVALID_ACTION

        # Reward for closing trades within the trade ttl
        if self.trade.is_open and action.action_type is ActionType.CLOSE:
            # Reward agent for closing a profitable scalp
            profitable_scalp_mask = (ttls >= GPUConstants.DEFAULT_TRADE_TTL - 5) & (trade_profits > self.trade.transaction_fee)
            reward_tensor += torch.sum(profitable_scalp_mask * GPUReward.TRADE_CLOSED_IN_PROFIT)

            # Punish agent for closing too early
            early_scalp_mask = (ttls >= GPUConstants.DEFAULT_TRADE_TTL - 5) & (trade_profits <= self.trade.transaction_fee)
            reward_tensor -= torch.sum(early_scalp_mask * GPUPunishment.CLOSING_TOO_QUICK)

            # Reward agent for closing trades within the trade ttl
            within_ttl_mask = (ttls > 0) & (ttls <= GPUConstants.DEFAULT_TRADE_TTL - 5)
            reward_tensor += torch.sum(within_ttl_mask * GPUReward.TRADE_CLOSED_WITHIN_TTL)

            # Only reward if closed within ttl and in profit
            profitable_within_ttl_mask = within_ttl_mask & (trade_profits > self.trade.transaction_fee)
            reward_tensor += torch.sum(profitable_within_ttl_mask * GPUReward.TRADE_CLOSED_IN_PROFIT)
            
            # Punish the agent for holding onto a losing trade for too long
            losing_trade_mask = (trade_profits < -(self.current_balance * 0.02)) & (ttls <= GPUConstants.DEFAULT_TRADE_TTL - 60)
            reward_tensor -= torch.sum(losing_trade_mask * GPUPunishment.HOLDING_LOSSING_TRADE)

            # Punish the agent for holding a trade with a big loss
            big_loss_mask = trade_profits < -significant_loss_threshold
            reward_tensor -= torch.sum(big_loss_mask * GPUPunishment.HOLDING_LOSSING_TRADE)

            # Punish the agent for holding a trade with too big a profit
            big_profit_mask = trade_profits > significant_gain_theshold
            reward_tensor -= torch.sum(big_profit_mask * GPUPunishment.RISKY_HOLDING)

            # Punish for holding trades too long
            holding_too_long_mask = ttls < 0
            reward_tensor -= torch.sum(holding_too_long_mask * GPUPunishment.HOLDING_TRADE_TOO_LONG)

            # Punish for holding trades WAY too long
            holding_way_too_long_mask = ttls < -ApplicationConstants.TRADE_TTL_OVERDRAFT_LIMIT
            reward_tensor -= torch.sum(holding_way_too_long_mask * GPUPunishment.TRADE_HELD_TOO_LONG)

        # Reward for doing nothing if no significant loss
        if action.action_type is ActionType.DO_NOTHING:
            if not self.trade.is_open or trade_profits.max() >= -significant_loss_threshold:
                reward_tensor += GPUReward.SMALL_REWARD_FOR_DOING_NOTHING
        
        # Punish agent if trade window expired
        if self.trade_window <= 0:
            reward_tensor -= GPUPunishment.NO_TRADE_OPEN
        
        self.reward = reward_tensor.item()
        return self.reward


    def _calc_sum_margin(self) -> float:
        return self.trade.get_margin().item()

    def _get_unrealized_pnl(self) -> float:
        trade_profits = self.trade.calculate_pnl(self.current_price)
        return trade_profits.sum().item()

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
                1 if self.trade.is_open else 0,
                ], dtype=torch.float32, device=self.device)
        return observation.cpu().numpy()


def _calculate_agent_improvement(average_win, average_loss, times_won, trades_closed) -> float:
    #TODO Move to GPU

    reward: float = 0.0
    max_array_size: int = 12_000

    # Truncate arrays if they exceed max array length
    for key in ['win_rate', 'average_win', 'average_loss', 'win_lose_ratio']:
        if agent_improvement_metric[key].size(0) > max_array_size:
            agent_improvement_metric[key] = agent_improvement_metric[key][max_array_size // 2:]


    # Calculate win rate
    win_rate = (times_won / trades_closed) if trades_closed > 0 else 0
    agent_improvement_metric['win_rate'] = torch.cat((agent_improvement_metric['win_rate'], torch.tensor([win_rate], dtype=torch.float32, device=DEVICE)))

    # Calculate average win and loss
    agent_improvement_metric['average_win'] = torch.cat((agent_improvement_metric['average_win'], torch.tensor([average_win], dtype=torch.float32, device=DEVICE)))
    agent_improvement_metric['average_loss'] = torch.cat((agent_improvement_metric['average_loss'], torch.tensor([average_loss], dtype=torch.float32, device=DEVICE)))
    # Calculate average win and loss
    win_lose_ratio = (average_win / abs(average_loss)) if abs(average_loss) > 0 else 1
    agent_improvement_metric['win_lose_ratio'] = torch.cat((agent_improvement_metric['win_lose_ratio'], torch.tensor([win_lose_ratio], dtype=torch.float32, device=DEVICE)))


    # Vectorized reward calculation
    win_lose_ratio_improved = (agent_improvement_metric['win_lose_ratio'][-1] > agent_improvement_metric['win_lose_ratio'][:-1].mean()).float()
    win_rate_improved = (agent_improvement_metric['win_rate'][-1] > agent_improvement_metric['win_rate'][:-1].mean()).float()
    average_win_improved = (agent_improvement_metric['average_win'][-1] > agent_improvement_metric['average_win'][:-1].mean()).float()

    reward += GPUReward.AGENT_IMPROVED * win_lose_ratio_improved
    reward += GPUReward.AGENT_IMPROVED * win_rate_improved
    reward += GPUReward.AGENT_IMPROVED * average_win_improved

    reward -= GPUPunishment.AGENT_NOT_IMPROVING * (1 - win_lose_ratio_improved)
    reward -= GPUPunishment.NO_TRADE_OPEN * (agent_improvement_metric['win_rate'][-1] == 0).float()
    reward -= GPUPunishment.AGENT_NOT_IMPROVING * (1 - win_rate_improved)
    reward -= GPUPunishment.AGENT_NOT_IMPROVING * (1 - average_win_improved)

    return reward
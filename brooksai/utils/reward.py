from typing import Dict, List

import torch

from brooksai.models import trade
from brooksai.models.action import Action
from brooksai.utils.action import ActionApply
from brooksai.models.constants import Punishment, Reward, ApplicationConstants, ActionType

class RewardFunction:
    """
    This class is responsible for calculating the reward for the agent.
    There are two separate reward systems

    Dense Reward: given at every step
        - monitors the agent's actions
        - monitors the agent's activity
    Sparse Reward: given every n hours
        - monitors the agent's overall performance
        - monitors the market sentiment and agent's actions to them

    """

    significant_loss_threshold: float = -80.0
    significant_gain_threshold: float = 150.0

    maximum_reward_per_step: float = 1
    minimum_reward_per_step: float = -1

    sparse_reward_interval: int = 180

    agent_improvement_metric: Dict[str, torch.Tensor] = {
        'win_rate': torch.tensor([], dtype=torch.float32),
        'average_win': torch.tensor([], dtype=torch.float32),
        'average_loss': torch.tensor([], dtype=torch.float32),
        'win_lose_ratio': torch.tensor([], dtype=torch.float32),
        'steps': torch.tensor([], dtype=torch.float32)
    }

    previous_actions: List[ActionType] = []
    previous_actions_for_sparse_reward: List[ActionType] = []
    previous_price: float = None

    @staticmethod
    def get_reward(action: Action, current_price: float, current_step: int):

        reward: float = 0.0
        if current_step % RewardFunction.sparse_reward_interval == 0 and current_step != 0:
            reward += RewardFunction.get_sparse_reward(current_price)

        reward += RewardFunction.get_dense_reward(action, current_price)
        return RewardFunction.normalize_reward(reward)


    @staticmethod
    def get_dense_reward(action: Action, current_price: float) -> float:
        """
        Calculate the reward for the agent per step
        :param action: Action: Action taken by the agent
        :param current_price: float: Current price of the asset
        :return: float: Reward
        """

        RewardFunction.previous_actions.append(action.action_type)
        RewardFunction.previous_actions_for_sparse_reward.append(action.action_type)

        reward: float = 0.0

        is_trade_open: bool = len(trade.open_trades) > 0
        trade_profit: float = trade.get_trade_profit(trade.open_trades[0], current_price) if is_trade_open else 0
        ttl: int = trade.open_trades[0].ttl if is_trade_open else 0

        invalid_trade: bool = is_trade_open and action.action_type in [ActionType.LONG, ActionType.SHORT]
        invalid_close: bool = not is_trade_open and action.action_type == ActionType.CLOSE

        if invalid_trade or invalid_close:
            # Punish for invalid action
            if invalid_close:
                ActionApply.increment_action_tracker('invalid_close', 1)
            return Punishment.INVALID_ACTION * -1

        if action.action_type == ActionType.DO_NOTHING:
            duration_of_inactivity = RewardFunction.previous_actions.count(ActionType.DO_NOTHING)
            reward += RewardFunction.do_nothing_curve(duration_of_inactivity) if not is_trade_open else 0

            # Check if a trade is open
            if is_trade_open:
                # Check if the trade profit is significant (bad thing)
                if trade_profit < RewardFunction.significant_loss_threshold:
                    reward -= Punishment.SIGNIFICANT_LOSS
                elif trade_profit > RewardFunction.significant_gain_threshold:
                    reward -= Punishment.SIGNIFICANT_LOSS

            # Clip reward to be within the range
            reward = max(reward, RewardFunction.minimum_reward_per_step)

        elif action.action_type == ActionType.CLOSE:

            RewardFunction.previous_actions = []

            # Only reward the agent for closing a trade if it made somewhat sense

            # Check if closed way too soon approx. 5 minutes
            if ttl > ApplicationConstants.DEFAULT_TRADE_TTL - 5:
                reward -= Punishment.CLOSING_TRADE_TOO_QUICKLY

            # Check if trade was in profit
            if trade_profit > ApplicationConstants.TRANSACTION_FEE:
                reward += Reward.TRADE_CLOSED_IN_PROFIT
            else:
                reward -= Punishment.TRADE_CLOSED_IN_LOSS

        else: # action.action_type in [ActionType.LONG, ActionType.SHORT]
            RewardFunction.previous_actions = []

            # Reward for opening a trade
            reward += Reward.TRADE_OPENED

        return reward

    @staticmethod
    def get_sparse_reward(current_price: float) -> float:
        """
        Calculate the reward for the agent per hour.
        The reward is calculated via the market sentiment and the agent's overall performance.

        :param current_price: float: Current price of the asset
        :return: float: Reward
        """
        if RewardFunction.previous_price is None:
            RewardFunction.previous_price = current_price

        reward: float = 0.0

        average_win = float(
                ActionApply.get_action_tracker('total_won')) / float(ActionApply.get_action_tracker('trades_closed')
                ) if ActionApply.get_action_tracker('trades_closed') > 0 else 0
        average_loss = float(
            ActionApply.get_action_tracker('total_lost')) / float(ActionApply.get_action_tracker('trades_closed')
            ) if ActionApply.get_action_tracker('trades_closed') > 0 else 0

        times_won = ActionApply.get_action_tracker('times_won')
        trades_closed = ActionApply.get_action_tracker('trades_closed')

        win_rate: float = (times_won / trades_closed) if trades_closed > 0 else 0
        win_loss_ratio = average_win / abs(average_loss) if abs(average_loss) > 0 else 1

        RewardFunction.agent_improvement_metric['win_rate'] = torch.cat(
            (RewardFunction.agent_improvement_metric['win_rate'], torch.tensor([win_rate]))
        )

        RewardFunction.agent_improvement_metric['win_lose_ratio'] = torch.cat(
            (RewardFunction.agent_improvement_metric['win_lose_ratio'], torch.tensor([win_loss_ratio]))
        )

        win_lose_ratio_improved = (
            RewardFunction.agent_improvement_metric['win_lose_ratio'][-1] > \
                RewardFunction.agent_improvement_metric['win_lose_ratio'][:-1].mean()
        ).float().item()

        win_rate_improved = (
            RewardFunction.agent_improvement_metric['win_rate'][-1] > \
                RewardFunction.agent_improvement_metric['win_rate'][:-1].mean()
        ).float().item()

        reward += Reward.AGENT_IMPROVED * win_lose_ratio_improved
        reward -= Punishment.AGENT_NOT_IMPROVING * (1 - win_lose_ratio_improved)

        reward += Reward.AGENT_IMPROVED * win_rate_improved
        reward -= Punishment.AGENT_NOT_IMPROVING * (1 - win_rate_improved)

        RewardFunction.previous_price = current_price
        RewardFunction.previous_actions_for_sparse_reward.clear()

        return reward

    @staticmethod
    def do_nothing_curve(duration: int,
                         max_reward: float = 0.01,
                         min_punishment: float = minimum_reward_per_step,
                         midpoint: int = ApplicationConstants.DO_NOTHING_MIDPOINT,
                         steepness: float = 0.1) -> float:
        """
        Function to calculate the reward for doing nothing.
        It is an inverse sigmoid function that returns a reward based on the duration of inactivity.
        Longer the duration, the lesser the reward.
        :param duration: int: Duration of inactivity
        :param max_reward: float: Maximum reward for doing nothing
        :param min_punishment: float: Minimum punishment for doing nothing
        :param midpoint: int: Midpoint of the curve (duration at which the reward is approx. 0)
        :param steepness: float: Steepness of the curve
        :return: float: Reward for doing nothing
        """

        # Convert duration to tensor
        duration = torch.tensor(duration, dtype=torch.float32)
        return (max_reward - min_punishment) / (1+torch.exp(-steepness + duration - midpoint)) + min_punishment


    @staticmethod
    def normalize_reward(reward: float) -> float:
        """
        Normalize the reward to be between -1 and 1.
        """
        reward = reward.item() if isinstance(reward, torch.Tensor) else reward

        top_part = reward - RewardFunction.minimum_reward_per_step
        bottom_part = RewardFunction.maximum_reward_per_step - RewardFunction.minimum_reward_per_step

        return  2 * ((top_part) / (bottom_part)) - 1

    @staticmethod
    def reset_rewards():
        """
        Reset rewards function, run per epsiode
        Steps are not reset as they are used to calculate the sparse reward
        """

        RewardFunction.agent_improvement_metric['win_rate'] = torch.tensor([], dtype=torch.float32)
        RewardFunction.agent_improvement_metric['average_win'] = torch.tensor([], dtype=torch.float32)
        RewardFunction.agent_improvement_metric['average_loss'] = torch.tensor([], dtype=torch.float32)
        RewardFunction.agent_improvement_metric['win_lose_ratio'] = torch.tensor([], dtype=torch.float32)

        RewardFunction.previous_actions = []
        RewardFunction.previous_actions_for_sparse_reward = []
        RewardFunction.previous_price = None

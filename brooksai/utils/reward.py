from typing import Dict, List

import torch

from brooksai.models import trade
from brooksai.models.action import Action
from brooksai.models.constants import Punishment, Reward, ApplicationConstants, Fee, ActionType

class RewardFunction:
    """
    This class is responsible for calculating the reward for the agent.
    The reward is calculated on the action taken by the agent.
    """

    significant_loss_threshold: float = -80.0
    significant_gain_threshold: float = 80.0

    maximum_reward_per_step: float = 0.8
    minimum_reward_per_step: float = -0.8

    agent_improvement_metric: Dict[str, torch.Tensor] = {
        'win_rate': torch.tensor([], dtype=torch.float32),
        'average_win': torch.tensor([], dtype=torch.float32),
        'average_loss': torch.tensor([], dtype=torch.float32),
        'win_lose_ratio': torch.tensor([], dtype=torch.float32),
        'steps': torch.tensor([], dtype=torch.float32)
    }

    previous_actions: List[ActionType] = []

    @staticmethod
    def do_nothing_curve(duration: int,
                         max_reward: float = 0.01,
                         min_punishment: float = minimum_reward_per_step,
                         midpoint: int = 60,
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
    def calculate_reward(action: Action, current_price: float) -> float:
        """
        Calculate the reward based on the action taken by the agent.
        :param action: Action: Action taken by the agent
        :param current_price: float: Current price of the asset
        :return: float: Reward
        """

        # Track actions, to calculate reward for inactivity purposes
        RewardFunction.previous_actions.append(action.action_type)

        reward: float = 0.0

        is_trade_open: bool = len(trade.open_trades) > 0
        trade_profit: float = trade.get_trade_profit(trade.open_trades[0], current_price) if is_trade_open else 0
        ttl: int = trade.open_trades[0].ttl if is_trade_open else 0

        invalid_trade: bool = is_trade_open and action.action_type in [ActionType.LONG, ActionType.SHORT]
        invalid_close: bool = not is_trade_open and action.action_type == ActionType.CLOSE


        if invalid_trade or invalid_close:
            # Punish for invalid action
            return Punishment.INVALID_ACTION * -1


        if action.action_type == ActionType.DO_NOTHING:
            # Count the number of do nothing actions
            duration_of_inactivity = RewardFunction.previous_actions.count(ActionType.DO_NOTHING)
            reward += RewardFunction.do_nothing_curve(duration_of_inactivity) if not is_trade_open else 0

            # Punish for holding a trade with significant loss or gain
            if is_trade_open and trade_profit != 0:
                if trade_profit < RewardFunction.significant_loss_threshold:
                    reward -= Punishment.SIGNIFICANT_LOSS
                elif trade_profit > RewardFunction.significant_gain_threshold:
                    reward -= Punishment.SIGNIFICANT_GAIN

            # Clip reward to minimum reward
            if reward < RewardFunction.minimum_reward_per_step:
                return RewardFunction.minimum_reward_per_step

            return RewardFunction.normalize_reward(reward)

        elif action.action_type == ActionType.CLOSE:
            # Reset previous actions
            RewardFunction.previous_actions = []

            reward += Reward.CLOSE_TRADE

            # Provide bonus reward / punishment based on trade profit
            if trade_profit > Fee.TRANSACTION_FEE:
                reward += Reward.TRADE_CLOSED_IN_PROFIT
            else:
                reward -= Punishment.TRADE_CLOSED_IN_LOSS

            # Provide bonus reward if trade closed within TTL
            if 0 < ttl < ApplicationConstants.DEFAULT_TRADE_TTL - 5:
                reward += Reward.TRADE_CLOSED_WITHIN_TTL

        else: # action.action_type in [ActionType.LONG, ActionType.SHORT]
            RewardFunction.previous_actions = []
            reward += Reward.TRADE_OPENED

        return RewardFunction.normalize_reward(reward)


    @staticmethod
    def normalize_reward(reward: float) -> float:
        """
        Normalize the reward to be between -1 and 1.
        """
        reward = reward.item() if isinstance(reward, torch.Tensor) else reward
        return 2 * (
            reward - RewardFunction.minimum_reward_per_step
        ) / ( RewardFunction.maximum_reward_per_step - RewardFunction.minimum_reward_per_step ) - 1

    @staticmethod
    def calculate_agent_improvement(average_win, averager_loss, times_won, trades_closed, steps):
        """
        This method is run at the end of every run to evaluate how the agent preformed.
        """

        reward: float = 0.0

        win_rate = (times_won / trades_closed) if trades_closed > 0 else 0
        RewardFunction.agent_improvement_metric['win_rate'] = torch.cat(
            (RewardFunction.agent_improvement_metric['win_rate'], torch.tensor([win_rate]))
        )

        win_lose_ratio = (
            average_win / abs(averager_loss)
        ) if abs(averager_loss) > 0 else 1

        # Append average win, loss, steps, and win lose ratio to agent improvement metric
        RewardFunction.agent_improvement_metric['average_win'] = torch.cat(
            (RewardFunction.agent_improvement_metric['average_win'], torch.tensor([average_win]))
        )

        RewardFunction.agent_improvement_metric['average_loss'] = torch.cat(
            (RewardFunction.agent_improvement_metric['average_loss'], torch.tensor([averager_loss]))
        )

        RewardFunction.agent_improvement_metric['steps'] = torch.cat(
            (RewardFunction.agent_improvement_metric['steps'], torch.tensor([steps]))
        )

        RewardFunction.agent_improvement_metric['win_lose_ratio'] = torch.cat(
            (RewardFunction.agent_improvement_metric['win_lose_ratio'], torch.tensor([win_lose_ratio]))
        )

        # Vectorized operations to calculate reward
        win_lose_ratio_improved = (
            RewardFunction.agent_improvement_metric['win_lose_ratio'][-1] > \
                RewardFunction.agent_improvement_metric['win_lose_ratio'][:-1].mean()
        ).float().item()

        win_rate_improved = (
            RewardFunction.agent_improvement_metric['win_rate'][-1] > \
                RewardFunction.agent_improvement_metric['win_rate'][:-1].mean()
        ).float().item()

        average_win_improved = (
            RewardFunction.agent_improvement_metric['average_win'][-1] > \
                RewardFunction.agent_improvement_metric['average_win'][:-1].mean()
        ).float().item()

        better_average_steps = (
            RewardFunction.agent_improvement_metric['steps'][-1] > \
                RewardFunction.agent_improvement_metric['steps'][:-1].mean()
        ).float().item()

        if RewardFunction.agent_improvement_metric['steps'][:-1].numel() > 0:
            best_step = (
                RewardFunction.agent_improvement_metric['steps'][-1] > \
                    RewardFunction.agent_improvement_metric['steps'][:-1].max()
            ).float().item()
        else:
            best_step = 0

        # Calculate reward based on agent improvement
        reward += Reward.AGENT_IMPROVED * win_lose_ratio_improved
        reward -= Punishment.AGENT_NOT_IMPROVING * (1 - win_lose_ratio_improved)

        reward -= Punishment.NO_TRADE_OPEN * (
            RewardFunction.agent_improvement_metric['win_rate'][-1] == 0
        ).float().item()

        reward += Reward.AGENT_IMPROVED * win_rate_improved
        reward -= Punishment.AGENT_NOT_IMPROVING * (1 - win_rate_improved)

        reward += Reward.AGENT_IMPROVED * average_win_improved
        reward -= Punishment.AGENT_NOT_IMPROVING * (1 - average_win_improved)

        reward += Reward.AGENT_IMPROVED * better_average_steps
        reward -= Punishment.AGENT_NOT_IMPROVING * (1 - better_average_steps)

        reward += Reward.AGENT_IMPROVED * best_step

        return reward

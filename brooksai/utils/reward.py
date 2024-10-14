import torch

from brooksai.models import trade
from brooksai.models.action import Action
from brooksai.models.constants import Punishment, Reward, ApplicationConstants, Fee, ActionType

class RewardFunction:

    significant_loss_threshold = 80
    significant_gain_threshold = 80

    agent_improvement_metric = {
        'win_rate': torch.tensor([], dtype=torch.float32),
        'average_win': torch.tensor([], dtype=torch.float32),
        'average_loss': torch.tensor([], dtype=torch.float32),
        'win_lose_ratio': torch.tensor([], dtype=torch.float32),
        'steps': torch.tensor([], dtype=torch.float32)
    }

    @staticmethod
    def calculate_reward(action: Action, current_price, current_balance, trade_window):

        reward: float = 0.0

        is_trade_open = len(trade.open_trades) > 0
        trade_profit = trade.get_trade_profit(trade.open_trades[0], current_price) if is_trade_open else 0
        ttl = trade.open_trades[0].ttl if is_trade_open else 0

        invalid_trade = is_trade_open and action.action_type in [ActionType.LONG, ActionType.SHORT]
        invalid_close = not is_trade_open and action.action_type == ActionType.CLOSE


        if invalid_trade or invalid_close:
            reward -= Punishment.INVALID_ACTION
            return reward
        

        if is_trade_open and \
            (trade_profit < -RewardFunction.significant_loss_threshold or trade_profit > RewardFunction.significant_gain_threshold):
            reward -= Punishment.SIGNIFICANT_LOSS
            reward -= Punishment.RISKY_HOLDING
            return reward

        if action.action_type == ActionType.DO_NOTHING:
            reward += Reward.DO_NOTHING
            return reward
        
        if trade_window <= 0:
            reward -= Punishment.NO_TRADE_WITHIN_WINDOW
            return reward

        if action.action_type == ActionType.CLOSE:
            reward += Reward.CLOSE_TRADE

            if trade_profit > Fee.TRANSACTION_FEE:
                reward += Reward.TRADE_CLOSED_IN_PROFIT

            if 0 < ttl < ApplicationConstants.DEFAULT_TRADE_TTL - 5:
                reward += Reward.TRADE_CLOSED_WITHIN_TTL

            if ApplicationConstants.DEFAULT_TRADE_TTL - 10 < ttl < ApplicationConstants.DEFAULT_TRADE_TTL:
                if trade_profit < Fee.TRANSACTION_FEE:
                    reward -= Punishment.TRADE_CLOSED_IN_LOSS
                    reward -= Punishment.CLOSING_TOO_QUICK

                    return reward

            if trade_profit < Fee.TRANSACTION_FEE:
                reward -= Punishment.TRADE_CLOSED_IN_LOSS
                return reward

        if action.action_type in [ActionType.LONG, ActionType.SHORT]:
            reward += Reward.TRADE_OPENED

        return reward

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

        RewardFunction.agent_improvement_metric['average_win'] = torch.cat(
            (RewardFunction.agent_improvement_metric['average_win'], torch.tensor([average_win]))
        )

        RewardFunction.agent_improvement_metric['average_loss'] = torch.cat(
            (RewardFunction.agent_improvement_metric['average_loss'], torch.tensor([averager_loss]))
        )

        RewardFunction.agent_improvement_metric['steps'] = torch.cat(
            (RewardFunction.agent_improvement_metric['steps'], torch.tensor([steps]))
        )

        # Calculate average win and loss

        win_lose_ratio = (
            average_win / abs(averager_loss)
        ) if abs(averager_loss) > 0 else 1

        RewardFunction.agent_improvement_metric['win_lose_ratio'] = torch.cat(
            (RewardFunction.agent_improvement_metric['win_lose_ratio'], torch.tensor([win_lose_ratio]))
        )

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

        reward += Reward.AGENT_IMPROVED * win_lose_ratio_improved
        reward -= Punishment.AGENT_NOT_IMPROVING * (1 - win_lose_ratio_improved)

        reward -= Punishment.NO_TRADE_OPEN * (RewardFunction.agent_improvement_metric['win_rate'][-1] == 0).float().item()

        reward += Reward.AGENT_IMPROVED * win_rate_improved
        reward -= Punishment.AGENT_NOT_IMPROVING * (1 - win_rate_improved)

        reward += Reward.AGENT_IMPROVED * average_win_improved
        reward -= Punishment.AGENT_NOT_IMPROVING * (1 - average_win_improved)

        reward += Reward.AGENT_IMPROVED * better_average_steps
        reward -= Punishment.AGENT_NOT_IMPROVING * (1 - better_average_steps)

        # Give a big reward for the best step
        reward += Reward.AGENT_IMPROVED * best_step * 1000

        return reward

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

        invalid_trade = is_trade_open and action.action_type in [ActionType.LONG, ActionType.SHORT]
        invalid_close = not is_trade_open and action.action_type == ActionType.CLOSE

        # Check invalid actions
        if invalid_trade or invalid_close:
            reward -= Punishment.INVALID_ACTION

        #  Agent is in a trade
        if is_trade_open:
            # if the agent tries to close the trade
            if action.action_type == ActionType.CLOSE:
                # Reward for closing a trade
                reward += Reward.CLOSE_TRADE

                # check trade ttl
                ttl = trade.open_trades[0].ttl
                if ttl >= ApplicationConstants.DEFAULT_TRADE_TTL - 5:
                    # Reward agent for closing a profitable scalp
                    if trade_profit > Fee.TRANSACTION_FEE:
                        reward += Reward.TRADE_CLOSED_IN_PROFIT
                    # If scalp is not profitable, punish agent for closing too early
                    else:
                        reward -= Punishment.CLOSING_TOO_QUICK
                if 0 < ttl <= ApplicationConstants.DEFAULT_TRADE_TTL - 5:
                    reward += Reward.TRADE_CLOSED_WITHIN_TTL

                    if trade_profit > Fee.TRANSACTION_FEE:
                        reward += Reward.TRADE_CLOSED_IN_PROFIT

                if ttl < 0:
                    reward -= Punishment.HOLDING_TRADE_TOO_LONG

                if ttl < -ApplicationConstants.TRADE_TTL_OVERDRAFT_LIMIT:
                    reward -= Punishment.TRADE_HELD_TOO_LONG * 5

            # Punish the agent for holding onto a lossing trade for too long (1 hours)
            if trade_profit < -(current_balance * 0.02) and \
                trade.open_trades[0].ttl <= ApplicationConstants.DEFAULT_TRADE_TTL - 60:
                reward -= Punishment.HOLDING_LOSSING_TRADE
            # Punish the agent for holding a trade with a big loss
            if trade_profit < -RewardFunction.significant_loss_threshold:
                reward -= Punishment.HOLDING_LOSSING_TRADE
            # Reward for doing nothing
            elif action.action_type == ActionType.DO_NOTHING:
                reward += Reward.SMALL_REWARD_FOR_DOING_NOTHING
            # Punish the agent for holding a trade with a big profit
            if trade_profit > RewardFunction.significant_gain_threshold:
                reward -= Punishment.RISKY_HOLDING

        if trade_window <= 0:
            reward -= Punishment.NO_TRADE_OPEN

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

        best_step = (
            RewardFunction.agent_improvement_metric['steps'][-1] > \
                RewardFunction.agent_improvement_metric['steps'][:-1].max()
        ).float().item()

        reward += Reward.AGENT_IMPROVED * win_lose_ratio_improved
        reward -= Punishment.AGENT_NOT_IMPROVING * (1 - win_lose_ratio_improved)

        reward -= Punishment.NO_TRADE_OPEN * (RewardFunction.agent_improvement_metric['win_rate'][-1] == 0).float().item()

        reward += Reward.AGENT_IMPROVED * win_rate_improved
        reawrd -= Punishment.AGENT_NOT_IMPROVING * (1 - win_rate_improved)

        reward += Reward.AGENT_IMPROVED * average_win_improved
        reward -= Punishment.AGENT_NOT_IMPROVING * (1 - average_win_improved)

        reward += Reward.AGENT_IMPROVED * better_average_steps
        reward -= Punishment.AGENT_NOT_IMPROVING * (1 - better_average_steps)

        # Give a big reward for the best step
        reward += Reward.AGENT_IMPROVED * (best_step * 3 if best_step > 0 else 0)

        return reward

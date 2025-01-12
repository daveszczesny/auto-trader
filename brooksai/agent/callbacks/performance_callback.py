import os
import logging

from stable_baselines3.common.callbacks import BaseCallback

from brooksai.utils.action import ActionApply

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger('AutoTrader')

best_model_base_path: str = 'best_models/'

class EvaluatePerformanceCallback(BaseCallback):
    """
    A callback that evaluates the performance of the model at regular intervals.
    """
    def __init__(self, eval_env, eval_freq: int = 80_000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_performance = 700

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each training step.
        """
        if self.n_calls % self.eval_freq == 0:
            performance = evaluate_model(self.eval_env)
            if performance > self.best_performance:
                self.best_performance = performance

                no_best_models_saved = len([name for name in os.listdir(best_model_base_path) if os.path.isfile(os.path.join(best_model_base_path, name))])
                model_path = best_model_base_path + f'best_model_cycle_{no_best_models_saved+1}.zip'
                self.model.save(model_path)
                if self.verbose > 0:
                    logger.info(f'New best model saved with performance: {performance}')

        return True


def evaluate_model(env):
    """
    Evaluation of the model's performance in the previous run
    Taking into account the final balance, reward and any unrealized pnl
    """

    balance = env.get_attr('current_balance')[0]
    reward = env.get_attr('reward')[0]
    unrealized_pnl = env.get_attr('unrealised_pnl')[0]
    trades_placed = ActionApply.get_action_tracker('trades_opened')
    times_won = ActionApply.get_action_tracker('times_won')
    win_rate = times_won / trades_placed if trades_placed > 0 else 0

    phi = 1

    # if only one trade was placed the win rate could appear as 1, which is not accurate
    if trades_placed <= 1:
        phi = 0

    alpha = 0.7
    beta = 0.4
    gamma = 0.2
    delta = 0.4

    performance =  (alpha * balance) + (beta * reward) + (gamma * unrealized_pnl) + (delta * win_rate)
    performance = phi * performance # if no trades or one trade were placed the performance is 0

    logger.info(f'Evaluation results - balance: {balance}, reward: {reward}, unrealized_pnl: {unrealized_pnl}, win_rate: {win_rate}, trades placed: {trades_placed}')
    logger.info(f'Calculated performance metric: {performance}')

    return performance
import os
import logging
import json

from typing import Any, Dict

from stable_baselines3.common.callbacks import BaseCallback

from brooksai.config_manager import ConfigManager
from brooksai.utils.action import ActionApply

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger('AutoTrader')

config = ConfigManager()

best_model_base_path: str = 'best_models/'

DEFAULT_PERFORMANCE_BENCHMARK = 1400
METADATA_PATH = os.path.join(best_model_base_path, 'metadata.json')


# Ensure metadata path exists
os.makedirs(best_model_base_path, exist_ok=True)

# Initialize the JSON file with an empty list if it doesn't exist or is empty
if not os.path.exists(METADATA_PATH) or os.stat(METADATA_PATH).st_size == 0:
    with open(METADATA_PATH, 'w') as f:
        json.dump([], f, indent=4)

class EvaluatePerformanceCallback(BaseCallback):
    """
    A callback that evaluates the performance of the model at regular intervals.
    """
    def __init__(self, eval_env: Any, eval_freq: int = 80_000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_performance = self.load_best_performance_model()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each training step.
        This method also controls the strategy the AI agent will use.
        The model can pick between three strategies: profit-seeking, risk-aversed, and balanced.
        These strategies differentiate in the way they calculate the performance metric.
        """
        if self.n_calls % self.eval_freq == 0:
            performance = evaluate_model(
                self.eval_env,
                **get_strategy_params(
                    config.get('performance_callback.selected_strategy')
                    )
            )
            if performance > self.best_performance:
                self.best_performance = performance

                no_best_models_saved = len([name for name in os.listdir(best_model_base_path) \
                        if os.path.isfile(os.path.join(best_model_base_path, name)) and \
                            name.endswith('.zip')])
                    
                model_path = best_model_base_path + f'best_model_cycle_{no_best_models_saved+1}.zip'
                self.model.save(model_path)
                self.save_metadata(model_path, performance)
                if self.verbose > 0:
                    logger.info(f'New best model saved with performance: {performance}')

        return True

    def load_best_performance_model(self) -> float:
        """
        Find best performance metric from the metadata file
        """
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
                return max((item['performance'] for item in metadata), default=DEFAULT_PERFORMANCE_BENCHMARK)
        return DEFAULT_PERFORMANCE_BENCHMARK


    def save_metadata(self, model_path: str, performance: float) -> None:
        """
        Save model and performance to metadata file
        """
        metadata = []
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
                if isinstance(metadata, dict):
                    metadata = [metadata]
        metadata.append({'model_path': model_path, 'performance': performance})
        with open(METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=4)


def evaluate_model(env: Any, **kwargs) -> float:
    """
    Evaluation of the model's performance in the previous run
    Taking into account the final balance, reward and any unrealized pnl

    :param env: The environment to evaluate
    :param kwargs: The strategy to use for evaluation
    :return: The performance metric
    """

    weight_balance = kwargs.get('weight_balance', 0.5)
    weight_reward = kwargs.get('weight_reward', 0.8)
    weight_upnl = kwargs.get('weight_upnl', 0.3)
    weight_winrate = kwargs.get('weight_winrate', 0.5)
    weight_avgloss = kwargs.get('weight_avgloss', 1.2)
    weight_invalidclose = kwargs.get('weight_invalidclose', 1.0)

    balance = env.get_attr('current_balance')[0]
    reward = env.get_attr('reward')[0]
    unrealized_pnl = env.get_attr('unrealised_pnl')[0]
    trades_placed = ActionApply.get_action_tracker('trades_opened')
    times_won = ActionApply.get_action_tracker('times_won')
    total_lost = ActionApply.get_action_tracker('total_lost')
    invalid_close_count = ActionApply.get_action_tracker('invalid_close')
    avg_loss = total_lost / trades_placed if trades_placed > 0 else 0
    win_rate = times_won / trades_placed if trades_placed > 0 else 0

    phi = 1 if trades_placed > 1 else 0

    positive_clause =  (weight_balance * balance) + (weight_reward * reward) + (weight_upnl * unrealized_pnl) + (weight_winrate * win_rate)
    negative_clause = (avg_loss * weight_avgloss) + (invalid_close_count * weight_invalidclose)
    performance = (phi * positive_clause) - negative_clause

    normalizetion_factor = sum(abs(value) for value in [
        weight_balance,
        weight_reward,
        weight_upnl,
        weight_winrate])

    if normalizetion_factor <= 0:
        raise ValueError('Normalization factor should be greater than 0')

    performance = performance / normalizetion_factor

    logger.info(f'Evaluation results - balance: {balance}, reward: {reward}, unrealized_pnl: {unrealized_pnl}, win_rate: {win_rate}, trades placed: {trades_placed}')
    logger.info(f'Calculated performance metric: {performance}')

    return performance



def get_strategy_params(strategy: str) -> Dict[str, float]:
    strategies = config.get('performance_callback.strategies')
    if strategies is None:
        raise ValueError('No strategies found in the config file')
    for strat in strategies:
        if strat['strategy']['name'] == strategy.lower():
            return strat['strategy']
        
    raise ValueError(f'No strategy found with the name {strategy.lower()}')

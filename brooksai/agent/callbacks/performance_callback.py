import os
import logging
import json

from typing import Any, Dict

from stable_baselines3.common.callbacks import BaseCallback

from brooksai.utils.action import ActionApply

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger('AutoTrader')

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
            performance = evaluate_model(self.eval_env, **get_strategy_params('risk-aversed'))
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

    alpha = kwargs.get('alpha', 0.5)
    beta = kwargs.get('beta', 0.8)
    gamma = kwargs.get('gamma', 0.3)
    delta = kwargs.get('delta', 0.5)
    zeta = kwargs.get('zeta', 1.2)

    balance = env.get_attr('current_balance')[0]
    reward = env.get_attr('reward')[0]
    unrealized_pnl = env.get_attr('unrealised_pnl')[0]
    trades_placed = ActionApply.get_action_tracker('trades_opened')
    times_won = ActionApply.get_action_tracker('times_won')
    total_lost = ActionApply.get_action_tracker('total_lost')
    avg_loss = total_lost / trades_placed if trades_placed > 0 else 0
    win_rate = times_won / trades_placed if trades_placed > 0 else 0

    phi = 1 if trades_placed > 1 else 0

    positive_clause =  (alpha * balance) + (beta * reward) + (gamma * unrealized_pnl) + (delta * win_rate)
    negative_clause = avg_loss * zeta
    performance = (phi * positive_clause) - negative_clause

    normalizetion_factor = alpha + beta + gamma + delta + zeta
    if normalizetion_factor <= 0:
        raise ValueError('Normalization factor should be greater than 0')
    
    performance = performance / normalizetion_factor

    logger.info(f'Evaluation results - balance: {balance}, reward: {reward}, unrealized_pnl: {unrealized_pnl}, win_rate: {win_rate}, trades placed: {trades_placed}')
    logger.info(f'Calculated performance metric: {performance}')

    return performance



def get_strategy_params(strategy: str) -> Dict[str, float]:
    strategies = {
        # More aggressive strategy, focusing on maximizing profits, even at the cost of higher risk
        'profit-seeking': {
            'alpha': 0.6,
            'beta': 1.0,
            'gamma': 0.4,
            'delta': 0.4,
            'zeta': 1.0,
        },
        # More conservative strategy, focusing on minimizing risk, even at the cost of lower profits
        'risk-aversed': {
            'alpha': 0.4,
            'beta': 0.6,
            'gamma': 0.2,
            'delta': 0.6,
            'zeta': 1.5,
        },
        # Balanced strategy, focusing on a balance between profits and risk
        'balanced': {
            'alpha': 0.5,
            'beta': 0.8,
            'gamma': 0.3,
            'delta': 0.5,
            'zeta': 1.2,
        }
    }
    return strategies.get(strategy.lower(), {})

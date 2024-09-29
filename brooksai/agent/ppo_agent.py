import gymnasium as gym
from stable_baselines3 import PPO

from typing import Optional

from env.models.constants import ActionType, action_type_mapping
from env.models.action import Action, TradeAction
from env.models.trade import get_trade_by_id, open_trades, Trade

class PPOAgent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.model = PPO('MlpPolicy', env, verbose=1)

    def learn(self, total_timesteps: int = 4_000_000):
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path)

    def predict(self, observation):
        raw_action, _ = self.model.predict(observation)
        return raw_action
    

    def _convert_to_action(self, raw_action) -> Action:

        action_type = action_type_mapping.get(raw_action[0])
        action: Action

        # If the action is to open a trade
        if action_type in [ActionType.LONG, ActionType.SHORT]:
            lot_size: float = round(float(raw_action[1]), 2)
            stop_loss: Optional[float] = float(raw_action[2]) if raw_action[2] != -1 else None
            take_profit: Optional[float] = float(raw_action[3]) if raw_action[3] != -1 else None

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
        # If the action is to close
        elif action_type == ActionType.CLOSE:
            trade_index = int(raw_action[4])
            trade: Optional[Trade] = None
            if 0 <= trade_index < len(open_trades) :
                trade = open_trades.get(trade_index, None)

            if trade == None:
                action = Action(action_type=ActionType.DO_NOTHING)
            else:
                action = Action(action_type=action_type, trade=trade)
        else:
            action = Action(action_type=action_type)

        return action

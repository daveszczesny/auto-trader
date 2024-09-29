from typing import Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from currency_converter import CurrencyConverter

from brooksai.env.models.trade import reset_open_trades, get_trade_state, close_trade, \
    trigger_stop_or_take_profit, close_all_trades, Trade, open_trades
from brooksai.env.models.constants import TradeType, ActionType, Punishment, Fee, \
    DEFAULT_TRADE_WINDOW, action_type_mapping, MAX_TRADES
from brooksai.env.models.action import Action, TradeAction
from brooksai.env.utils.converter import pip_to_profit

c = CurrencyConverter()

class ForexEnv(gym.Env):
    """
    Custom Environment for forex trading
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, data: str, initial_balance: int = 1_000, render_mode: Optional[Any] = None):

        self._load_data(data)

        # environment variables
        self.n_steps = len(self.data)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low = 0,
                                            high = np.inf,
                                            shape=(8 + MAX_TRADES,),
                                            dtype=np.float32)

        # action taken, lot size, stop loss, take profit, trade index to close
        self.action_space = spaces.Box(
            low = np.array([0.0, 0.01, -1.0, -1.0, 0.0], dtype=np.float32),
            high = np.array([1.0, 1.0, 300.0, 300.0, MAX_TRADES - 1], dtype=np.float32),
            dtype=np.float32
        )

        self.current_step: int = 0
        self.current_dataset: pd.Series = self.data.iloc[self.current_step]
        self.current_price: float = self.current_dataset['bid_close']
        self.reward: float = 0.0
        self.trade_window: int = DEFAULT_TRADE_WINDOW

        # agent variables
        self.initial_balance: float = initial_balance
        self.current_balance: float = initial_balance
        self.unrealized_profit: float = 0.0
        self.previous_balance: float = 0.0


    def step(self, action: np.ndarray) -> Tuple[np.array, float, bool, bool, dict]:
        """
        Execute one time step within the environment
        :param action: The action taken by the agent. (0: do nothing, 1: long, 2: short, 3: close)
        :return new observation, reward, done, info
        """
        print(action)
        action: Action = self.construct_action(action)

        with open('brooksai_logs.txt', 'a') as f:
            f.write(f"Action: {action.action_type}\n")

        self.reward = 0.0
        self.previous_balance = self.current_balance
        self.current_price = self.data.iloc[self.current_step]['bid_close']

        # Check for stop loss or take profits
        trigger_stop_or_take_profit(self.current_price)

        self.apply_actions(action)

        self.unrealized_profit = sum(
            pip_to_profit(self.current_price - trade.open_price, trade.lot_size) if
            trade.trade_type == TradeType.LONG else
            pip_to_profit(trade.open_price - self.current_price,
                          trade.lot_size) for trade in open_trades
        )


        self.apply_environment_rules()

        # proceed to the next time step
        done = False
        self.current_step += 1
        if self.current_step >= self.n_steps:
            self.reward += 100
            done = True
        elif self.current_balance <= 0:
            self.reward -= 100
            done = True

        if self.current_step % 500 == 0:
            with open('brooksai_logs.txt', 'a') as f:
                f.write(f"Final balance: {round(self.current_balance, 2)}\n")

        if done:
            self.current_balance += close_all_trades(self.current_price)

        # Calculate reward
        reward = self.current_balance + self.unrealized_profit + self.reward

        return self._get_observation(), reward, done, False, {}

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.array, dict]:
        """
        Reset the state of the environment to an initial state
        """
        super().reset(seed=seed)

        with open('brooksai_logs.txt', 'a') as f:
            f.write(f"\n\nBalance: {round(self.current_balance, 2)}\n")
            f.write(f"Unrealized Profit: {round(self.unrealized_profit, 2)}\n")
            f.write(f"Open Trades: {len(open_trades)}\n\n\n")

        # reset environment variables
        self.current_step = 0
        self.current_dataset = self.data.iloc[self.current_step]
        self.current_price = self.current_dataset['bid_close']
        self.trade_window = DEFAULT_TRADE_WINDOW
        self.reward = 0.0
        reset_open_trades()

        # reset agent variables
        self.current_balance = self.initial_balance
        self.unrealized_profit = 0.0

        return self._get_observation(), {}

    def render(self, mode: str ='human') -> None:
        """
        Render the environment to the screen
        """
        if mode == 'human':
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.current_balance}')
            print(f'Unrealized Profit: {self.unrealized_profit}')
            print(f'Open Trades: {len(open_trades)}')
            print(f'Price: {self.current_price}')

    def construct_action(self, raw_action: np.ndarray) -> Action:
        action_type = action_type_mapping.get(int(raw_action[0] * 3), ActionType.DO_NOTHING)

        action: Action

        if action_type in [ActionType.LONG, ActionType.SHORT]:
            if raw_action[1] is None:
                action = Action(action_type=ActionType.DO_NOTHING)
                return action

            lot_size: float = round(float(raw_action[1]), 2)
            stop_loss: Optional[float] = float(raw_action[2]) if raw_action[2] != -1 else None
            take_profit: Optional[float] = float(raw_action[3]) if raw_action[3] != -1 else None

            # Penalise for setting small stop loss or take profit
            if stop_loss and stop_loss < 5:
                self.reward -= Punishment.NO_STOP_LOSS.value
            if take_profit and take_profit < 5:
                self.reward -= Punishment.NO_TAKE_PROFIT.value

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
        elif action_type == ActionType.CLOSE:
            trade_index = int(raw_action[4])
            trade: Optional[Trade] = None
            if 0 <= trade_index < len(open_trades):
                trade = open_trades[trade_index]

            if trade is None:
                action = Action(action_type=ActionType.DO_NOTHING)
            else:
                action = Action(action_type=action_type, trade=trade)
        else:
            action = Action(action_type=action_type)

        return action

    def apply_actions(self, action: Action) -> None:
        """
        Apply the actions taken by the agent
        """

        if action.action_type == ActionType.DO_NOTHING:
            return

        if len(open_trades) < MAX_TRADES and \
            action.action_type in [ActionType.LONG, ActionType.SHORT]:
            Trade(
                lot_size=action.data.lot_size,
                open_price=self.current_price,
                trade_type=TradeType.LONG if action.action_type == ActionType.LONG
                                            else TradeType.SHORT,
                stop_loss=action.data.stop_loss,
                take_profit=action.data.take_profit
            )

            # Penalise for not setting stop loss or take profit
            if action.data.stop_loss is None:
                self.reward -= Punishment.NO_STOP_LOSS.value
            if action.data.take_profit is None:
                self.reward -= Punishment.NO_TAKE_PROFIT.value

        elif action.action_type == ActionType.CLOSE:
            self.current_balance += close_trade(action.trade, self.current_price)

    def apply_environment_rules(self) -> None:
        self._check_trade_ttl()

        # Check if margin call is needed
        # Margin call is triggered if the current balance is less than the sum
        # of the margin for all trades plus the unrealised profit
        if self.current_balance <= (self._calc_sum_margin() + self.unrealized_profit):
            self._margin_call()
            self.reward -= Punishment.MARGIN_CALLED.value

        self._check_trade_window()


    def _check_trade_ttl(self) -> None:
        """
        Deducts 1 minute from the time-to-live (TTL) of all active trades.
        Charges an extra day fee for each trade that remains open beyond its TTL.
        """
        for trade in open_trades:
            trade.ttl -= 1
            if trade.ttl <= 0:
                self.reward -= Fee.EXTRA_DAY_FEE.value

    def _margin_call(self) -> None:
        """
        Triggers margin call Event
        Closes all trades
        """
        if not open_trades:
            return

        value: float = 0.0
        for trade in open_trades:
            value += close_trade(trade, self.current_price)
            value -= Fee.TRANSACTION_FEE.value

        self.current_balance += value

    def _calc_sum_margin(self) -> float:
        """
        Calculate the sum of the margin for all trades
        """
        return sum(trade.get_margin() for trade in open_trades)

    def _check_trade_window(self) -> None:
        """
        Check if a trade was opened within the trade window
        """
        if self.trade_window == 0:
            self.reward -= Punishment.NO_TRADE_WITHIN_WINDOW.value
        elif self.trade_window < 0:
            self.trade_window += Punishment.NO_TRADE_WITHIN_WINDOW.value * self.trade_window

    def _get_observation(self) -> np.array:
        """
        Get the observation
        """

        trade_states = []
        for trade in open_trades:
            trade_state = get_trade_state(trade.uuid, self.current_price)
            trade_states.extend([
                trade_state['profit'],
            ])

        while len(trade_states) < MAX_TRADES:
            trade_states.append(0.0)


        return np.array([
            self.current_price,
            self.current_balance,
            self.unrealized_profit,
            len(open_trades),
            self.current_step,
            self.current_dataset['EMA_200'],
            self.current_dataset["EMA_50"],
            self.current_dataset["EMA_21"]
        ] + trade_states[:MAX_TRADES], dtype=np.float32)

    def _load_data(self, data: str) -> None:
        """
        Load the price data from resource file
        """

        self.data = pd.read_csv(data)

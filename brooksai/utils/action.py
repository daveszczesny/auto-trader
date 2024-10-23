import logging

import torch
import numpy as np
from brooksai.models.action import Action as ActionModel, TradeAction
from brooksai.models.constants import Fee, ActionType, TradeType, ApplicationConstants, action_type_mapping
from brooksai.models.trade import Trade, get_trade_profit, close_trade, open_trades


logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger('AutoTrader')

class ActionBuilder:

    @staticmethod
    def construct_action(raw_action: np.ndarray) -> ActionModel:
        """
        Construct the Agent action from raw action values
        :param raw_action: Raw action values
        :return: Action object
        """
        action_type = ActionBuilder._get_action_type(raw_action)

        if ActionBuilder._is_invalid_acition(action_type, raw_action):
            # If the action is invalid, do nothing
            return ActionModel(action_type=ActionType.DO_NOTHING)

        if action_type in [ActionType.LONG, ActionType.SHORT]:
            trade_action = ActionBuilder._create_trade_action(raw_action)
            return ActionModel(action_type=action_type, trade_data=trade_action)

        if action_type is ActionType.CLOSE:
            # This assumes that there is only one open trade at a time
            return ActionModel(action_type=action_type, trade=open_trades[0])

        return ActionModel(action_type=ActionType.DO_NOTHING)

    @staticmethod
    def _get_action_type(raw_action: torch.Tensor) -> ActionType:
        """
        Converts a continuous action value [0 - 1] to an ActionType
        :param raw_action: Raw action value
        :return: ActionType [DO_NOTHING, LONG, SHORT, CLOSE]
        """

        index = int(raw_action[0].item() * (len(action_type_mapping) - 1))
        return action_type_mapping.get(index, ActionType.DO_NOTHING)

    @staticmethod
    def _is_invalid_acition(action_type: ActionType, raw_action: torch.Tensor) -> bool:
        """
        Checks to see if the action is invalid for current environment state
        :param action_type: ActionType
        :param raw_action: Raw action values
        :return: True if invalid, False otherwise
        """
        invalid_long_or_short = action_type in [ActionType.LONG, ActionType.SHORT] and\
            raw_action[1].item() <= 0
        invalid_long_or_short_2 = action_type in [ActionType.LONG, ActionType.SHORT] and\
            len(open_trades) > 0
        invalid_long_or_short = invalid_long_or_short or invalid_long_or_short_2
        invalid_close = action_type is ActionType.CLOSE and len(open_trades) <= 0

        return invalid_long_or_short or invalid_close

    @staticmethod
    def _create_trade_action(raw_action: torch.Tensor) -> TradeAction:
        """
        Constructs the trade action from raw action values

        For the moment stop loss and take profit are not included

        :param raw_action: Raw action values
        :return: TradeAction object
        """
        lot_size = raw_action[1].item()

        # sl and tp will be added later
        stop_loss = None
        take_profit = None

        return TradeAction(
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size
        )


class ActionApply:

    action_tracker = {
        'trades_opened': 0,
        'trades_closed': 0,
        'total_won': 0,
        'total_lost': 0,
        'times_won': 0,
        'times_lost': 0,
    }

    @staticmethod
    def apply_action(action: ActionModel, **kwargs):
        """
        Apply the action to the environment
        :param action: Action to apply
        :param kwargs: Additional arguments
        :return: Profit or loss from the action
        """

        trade_window = kwargs.get('trade_window', None)
        current_price = kwargs.get('current_price', None)

        if trade_window is None:
            logger.warning("Trade window is None")
            trade_window = ApplicationConstants.DEFAULT_TRADE_WINDOW

        if current_price is None:
            logger.warning("Current price is None")
            return 0.0, trade_window

        if action.action_type in [ActionType.LONG, ActionType.SHORT]:
            """
            Open a LONG or SHORT trade
            """
            Trade(
                lot_size=action.trade_data.lot_size,
                open_price=current_price,
                trade_type=TradeType.LONG if action.action_type is ActionType.LONG else TradeType.SHORT
            )

            ActionApply.action_tracker['trades_opened'] += 1
            trade_window = ApplicationConstants.DEFAULT_TRADE_WINDOW

        elif action.action_type is ActionType.CLOSE:
            """
            Close existing trade
            """
            if not action.trade:
                return 0.0, trade_window

            ActionApply.action_tracker['trades_closed'] += 1
            value = get_trade_profit(action.trade, current_price) - Fee.TRANSACTION_FEE

            if value > 0:
                ActionApply.action_tracker['total_won'] += value
                ActionApply.action_tracker['times_won'] += 1
            else:
                ActionApply.action_tracker['total_lost'] += value
                ActionApply.action_tracker['times_lost'] += 1

            return close_trade(action.trade, current_price), trade_window

        else:
            # Action: DO NOTHING
            trade_window -= 1

        return 0.0, trade_window

    @staticmethod
    def get_action_tracker(key: str):
        return ActionApply.action_tracker.get(key, 0)

    @staticmethod
    def reset_tracker():
        ActionApply.action_tracker = {
            'trades_opened': 0,
            'trades_closed': 0,
            'total_won': 0,
            'total_lost': 0,
            'times_won': 0,
            'times_lost': 0,
        }

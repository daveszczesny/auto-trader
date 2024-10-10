import torch
from brooksai.models.action import Action as ActionModel, TradeAction
from brooksai.models.constants import Fee, ActionType, TradeType, action_type_mapping
from brooksai.models.trade import open_trades, Trade, get_trade_profit, close_trade

class ActionBuilder:

    @staticmethod
    def construct_action(raw_action: torch.Tensor) -> ActionModel:
        """
        Construct the Agent action from raw action values
        :param raw_action: Raw action values
        :return: Action object
        """
        action_type = ActionBuilder._get_action_type(raw_action)

        if ActionBuilder._is_invalid_acition(action_type, raw_action):
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

        current_price = kwargs.get('current_price', None)
        if current_price is None:
            print("Current price is None")
            return 0.0

        if action.action_type in [ActionType.LONG, ActionType.SHORT]:
            Trade(
                lot_size=action.trade_data.lot_size,
                open_price=current_price,
                trade_type=TradeType.LONG if action.action_type is ActionType.LONG else TradeType.SHORT
            )
            ActionApply.action_tracker['trades_opened'] += 1
        else:
            if not action.trade:
                return 0.0

            ActionApply.action_tracker['trades_closed'] += 1
            value = get_trade_profit(action.trade, current_price) - Fee.TRANSACTION_FEE
            if value > 0:
                ActionApply.action_tracker['total_won'] += value
                ActionApply.action_tracker['times_won'] += 1
            else:
                ActionApply.action_tracker['total_lost'] += value
                ActionApply.action_tracker['times_lost'] += 1

            return close_trade(action.trade, current_price)
        return 0.0

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

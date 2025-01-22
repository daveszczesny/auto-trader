from enum import Enum

from brooksai.config_manager import ConfigManager

config = ConfigManager()

class TradeType(Enum):
    LONG = config.get('environment.constants.trade_type.long', 1)
    SHORT = config.get('environment.constants.trade_type.short', -1)

class ActionType(Enum):
    DO_NOTHING = config.get('environment.constants.action.do_nothing', 0)
    LONG = config.get('environment.constants.action.long', 1)
    SHORT = config.get('environment.constants.action.short', 2)
    CLOSE = config.get('environment.constants.action.close', 3)


class Punishment:
    CLOSING_TRADE_TOO_QUICKLY = \
        config.get('environment.constants.punishment.closing_trade_too_quickly', 0.1)
    NO_TRADE_OPEN = \
        config.get('environment.constants.punishment.no_trade_open', 0.5)
    TRADE_CLOSED_IN_LOSS = \
        config.get('environment.constants.punishment.trade_closed_in_loss', 0.2)
    SIGNIFICANT_LOSS = \
        config.get('environment.constants.punishment.significant_loss', 0.2)
    INVALID_ACTION = \
        config.get('environment.constants.punishment.invalid_action', 1)
    AGENT_NOT_IMPROVING = \
        config.get('environment.constants.punishment.agent_not_improving', 0.3)

class Reward:
    TRADE_CLOSED_IN_PROFIT = \
        config.get('environment.constants.reward.trade_closed_in_profit', 0.5)
    TRADE_OPENED = \
        config.get('environment.constants.reward.trade_opened', 1.5)
    TRADE_CLOSED_WITHIN_TTL = \
        config.get('environment.constants.reward.trade_closed_within_ttl', 0.1)
    CLOSE_TRADE = \
        config.get('environment.constants.reward.close_trade', 0.2)
    AGENT_IMPROVED = \
        config.get('environment.constants.reward.agent_improved', 0.6)

action_type_mapping = {
    0: ActionType.DO_NOTHING,
    1: ActionType.LONG,
    2: ActionType.SHORT,
    3: ActionType.CLOSE
}

class ApplicationConstants:
    INITIAL_BALANCE = \
        config.get('environment.constants.application.initial_balance', 1_000)
    DEFAULT_TRADE_TTL = \
        config.get('environment.constants.application.ttl', 7200)
    DEFAULT_TRADE_WINDOW = \
        config.get('environment.constants.application.trade_window', 5_760)
    CONTRACT_SIZE = \
        config.get('environment.constants.application.contract_size', 100_000)
    LEVERAGE = \
        config.get('environment.constants.application.leverage', 500)
    TRANSACTION_FEE = \
        config.get('environment.constants.application.transaction_fee', 2.54)
    BIG_LOSS = \
        config.get('environment.constants.application.big_loss', 50)
    DEVICE = \
        config.get('environment.constants.application.device', 'cpu')

    DO_NOTHING_MIDPOINT = \
        config.get('environment.constants.application.do_nothing_midpoint', 2_880)

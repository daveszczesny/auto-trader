from enum import Enum

class TradeType(Enum):
    LONG = 1
    SHORT = -1

class ActionType(Enum):
    DO_NOTHING = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3


class Punishment:
    CLOSING_TRADE_TOO_QUICKLY = 0.1
    NO_TRADE_OPEN = 0.5
    TRADE_CLOSED_IN_LOSS = 0.5
    SIGNIFICANT_LOSS = 0.2
    INVALID_ACTION = 0.8
    AGENT_NOT_IMPROVING = 0.4

class Reward:
    TRADE_CLOSED_IN_PROFIT = 0.5
    TRADE_OPENED = 0.3
    TRADE_CLOSED_WITHIN_TTL = 0.1
    CLOSE_TRADE = 0.2
    AGENT_IMPROVED = 0.4

action_type_mapping = {
    0: ActionType.DO_NOTHING,
    1: ActionType.LONG,
    2: ActionType.SHORT,
    3: ActionType.CLOSE
}

class ApplicationConstants:
    INITIAL_BALANCE = 1_000
    DEFAULT_TRADE_TTL = 5_760 # 4 days
    DEFAULT_TRADE_WINDOW = 4_320 # 3 day
    CONTRACT_SIZE = 100_000
    LEVERAGE = 500
    TRANSACTION_FEE = 2.54
    BIG_LOSS = 50
    DEVICE = 'cpu'

    DO_NOTHING_MIDPOINT = 2_880 # 2 days

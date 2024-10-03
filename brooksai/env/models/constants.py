from enum import Enum

class TradeType(Enum):
    LONG = 1
    SHORT = -1

class ActionType(Enum):
    DO_NOTHING = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3


class Fee(Enum):
    TRANSACTION_FEE = 2.54
    EXTRA_DAY_FEE = 2.54

class Punishment:
    NO_STOP_LOSS = 2
    NO_TAKE_PROFIT = 2
    MARGIN_CALLED = 5
    NO_TRADE_WITHIN_WINDOW = 5
    INSUFFICIENT_MARGIN = 5
    INSUFFICIENT_FUNDS = 8
    MAX_TRADES_REACHED = 2

class Reward:
    TRADE_CLOSED = 1
    TRADE_CLOSED_IN_PROFIT = 3
    TRADE_OPENED = 1
    TRADE_CLOSED_WITHIN_TTL = 1
    COMPLETED_RUN = 10

action_type_mapping = {
    0: ActionType.DO_NOTHING,
    1: ActionType.LONG,
    2: ActionType.SHORT,
    3: ActionType.CLOSE
}


class ApplicationConstants:
    DEFAULT_TRADE_TTL = 14_400 # 10 days
    DEFAULT_TRADE_WINDOW = 4320 # 3 days
    CONTRACT_SIZE = 100_000
    LEVERAGE = 500
    MARGIN_LIMIT = 0.5
    MAX_TRADES = 10
    TP_AND_SL_SCALE_FACTOR = 200
    SIMPLE_MAX_TRADES = 1
    TRANSACTION_FEE = 2.54

    MINIMUM_REWARD = -73.54
    MAXIMUM_REWARD = 16

class Environments(Enum):
    DEV = "dev"
    PROD = "prod"
    TEST = "test"


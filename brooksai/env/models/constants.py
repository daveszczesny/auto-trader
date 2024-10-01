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

class Punishment(Enum):
    NO_STOP_LOSS = 20
    NO_TAKE_PROFIT = 15
    MARGIN_CALLED = 100
    NO_TRADE_WITHIN_WINDOW = 10
    INSUFFICIENT_MARGIN = 5

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

class Environments(Enum):
    DEV = "dev"
    PROD = "prod"
    TEST = "test"


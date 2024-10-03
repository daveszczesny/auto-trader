from enum import Enum

class TradeType(Enum):
    LONG = 1
    SHORT = -1

class ActionType(Enum):
    DO_NOTHING = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3


class Fee:
    TRANSACTION_FEE = 2.54
    EXTRA_DAY_FEE = 2

class Punishment:
    NO_STOP_LOSS = 2
    NO_TAKE_PROFIT = 2

    MARGIN_CALLED = 7
    NO_TRADE_WITHIN_WINDOW = 3
    INSUFFICIENT_MARGIN = 5
    INSUFFICIENT_FUNDS = 5
    MAX_TRADES_REACHED = 2
    NO_TRADE_OPEN = 1
    TRADE_CLOSED_IN_LOSS = 4.5
    MISSED_PROFIT = 2
    UNREALIZED_LOSS = 1
    SIGNIFICANT_LOSS=2

    INVALID_ACTION = 5

class Reward:
    TRADE_CLOSED = 1.2
    TRADE_CLOSED_IN_PROFIT = 10
    TRADE_OPENED = 0.5
    TRADE_CLOSED_WITHIN_TTL = 1.5
    COMPLETED_RUN = 4
    UNREALIZED_PROFIT = 2

    SMALL_REWARD_FOR_DOING_NOTHING = 0.1

action_type_mapping = {
    0: ActionType.DO_NOTHING,
    1: ActionType.LONG,
    2: ActionType.SHORT,
    3: ActionType.CLOSE
}


class ApplicationConstants:
    DEFAULT_TRADE_TTL = 7_200 # 5 days
    DEFAULT_TRADE_WINDOW = 4320 # 3 days
    TRADE_TTL_OVERDRAFT_LIMIT = -1440 # 1 day
    CONTRACT_SIZE = 100_000
    LEVERAGE = 500
    MARGIN_LIMIT = 0.5
    MAX_TRADES = 10
    TP_AND_SL_SCALE_FACTOR = 200
    SIMPLE_MAX_TRADES = 1
    TRANSACTION_FEE = 2.54

    BIG_LOSS = 50

class Environments(Enum):
    DEV = "dev"
    PROD = "prod"
    TEST = "test"


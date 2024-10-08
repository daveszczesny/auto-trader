from enum import Enum

import torch

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
    NO_TRADE_OPEN = 5
    TRADE_CLOSED_IN_LOSS = 4.5
    MISSED_PROFIT = 2
    UNREALIZED_LOSS = 1
    SIGNIFICANT_LOSS = 2

    INVALID_ACTION = 5
    HOLDING_TRADE_TOO_LONG = 2
    CLOSING_TOO_QUICK = 1.5
    HOLDING_LOSSING_TRADE = 2
    AGENT_NOT_IMPROVING = 5
    RISKY_HOLDING = 3

    TRADE_HELD_TOO_LONG = 20

class Reward:
    TRADE_CLOSED = 1.5
    TRADE_CLOSED_IN_PROFIT = 10
    TRADE_OPENED = 0.5
    COMPLETED_RUN = 4
    UNREALIZED_PROFIT = 2

    TRADE_CLOSED_WITHIN_TTL = 2
    SMALL_REWARD_FOR_DOING_NOTHING = 0.1
    BETTER_AVERAGE_TRADE = 0.1

    AGENT_IMPROVED = 10
    BETTER_AVERAGE_TRADE = 3


action_type_mapping = {
    0: ActionType.DO_NOTHING,
    1: ActionType.LONG,
    2: ActionType.SHORT,
    3: ActionType.CLOSE
}


class ApplicationConstants:
    DEFAULT_TRADE_TTL = 2_880 # 2 days
    DEFAULT_TRADE_WINDOW = 1440 # 1 day
    TRADE_TTL_OVERDRAFT_LIMIT = 7720 # 5 days
    CONTRACT_SIZE = 100_000
    LEVERAGE = 500
    MARGIN_LIMIT = 0.5
    MAX_TRADES = 10
    TP_AND_SL_SCALE_FACTOR = 200
    SIMPLE_MAX_TRADES = 1
    TRANSACTION_FEE = 2.54
    BIG_LOSS = 50
    DEVICE = 'cpu'

class Environments(Enum):
    DEV = "dev"
    PROD = "prod"
    TEST = "test"

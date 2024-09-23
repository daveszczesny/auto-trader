from enum import Enum


DATASET_NAME = 'training_data.csv'

class TimeConversion(Enum):
    ONE_DAY = 24 * 60,
    FIVE_DAYS = ONE_DAY * 5,
    MONTH_DAYS = FIVE_DAYS * 4,
    YEAR_DAYS = MONTH_DAYS * 12,
    ALL_DATA = 5_100_00,

class TradeType(Enum):
    LONG = 1,
    SHORT = 2,


class Decision(Enum):
    ENTER_TRADE = 1,
    CLOSE_TRADE = 2,
    DO_NOTHING = 3,

class TradeDecision(Enum):
    LONG = "LONG",
    SHORT = "SHORT",
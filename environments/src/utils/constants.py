from enum import Enum


DATASET_NAME = 'training_data.csv'

class TradeType(Enum):
    LONG = 1,
    SHORT = 2,


class Decision(Enum):
    LONG = 1,
    SHORT = 2,
    DO_NOTHING = 3,


class CloseDecision(Enum):
    CLOSE = 1
    DO_NOTHING = 2
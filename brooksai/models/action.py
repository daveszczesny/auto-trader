from dataclasses import dataclass
from typing import Optional

from brooksai.models.trade import Trade
from brooksai.models.constants import ActionType

@dataclass
class TradeAction:
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    lot_size: float = 0.01

@dataclass
class Action:
    action_type: ActionType
    trade_data: Optional[TradeAction] = None
    trade: Optional[Trade] = None

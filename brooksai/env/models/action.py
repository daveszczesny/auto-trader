from dataclasses import dataclass
from typing import Optional

from env.models.constants import ActionType
from env.models.trade import Trade

@dataclass
class TradeAction:
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    lot_size: float = 0.01


@dataclass
class Action:
    action_type: ActionType  # never null
    data: Optional[TradeAction] = None  # populated when opening trade
    trade: Optional[Trade] = None  # populated when closing trade
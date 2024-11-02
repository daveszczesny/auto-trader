import torch
import numpy as np
from enum import Enum

class ActionType(Enum):
    DO_NOTHING = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3



action_type_mapping = {
    0: ActionType.DO_NOTHING,
    1: ActionType.LONG,
    2: ActionType.SHORT,
    3: ActionType.CLOSE
}

def get_action_type(action: torch.Tensor) -> ActionType:
    index = int(action[0].item() * (len(action_type_mapping) - 1))
    return action_type_mapping.get(index, ActionType.DO_NOTHING)


def construct_action(action: np.ndarray) -> ActionType:
    action_type = get_action_type(action)

    response = {'action': action_type.name}

    if action_type in [ActionType.LONG, ActionType.SHORT]:
        response.update({
            'lot_size': action[1].item(),
            'stop_loss': None,
            'take_profit': None
        })

    return response


"""
This module contains the GPUTrade class, which is used to represent a trade on the GPU

It is also a one trade per GPU model, meaning that only one trade can be opened on a GPU at a time
"""

import torch

from currency_converter import CurrencyConverter

from brooksai.env.models.constants import ApplicationConstants

c = CurrencyConverter()

class GPUTrade:
    
    ttl: torch.Tensor = torch.tensor(ApplicationConstants.DEFAULT_TRADE_TTL, dtype=torch.float32)
    _lot_size: torch.Tensor
    _open_price: torch.Tensor
    _trade_type: torch.Tensor
    stop_loss: torch.Tensor
    take_profit: torch.Tensor
    transaction_fee: torch.Tensor

    def __init__(self):
            self.device = torch.device(ApplicationConstants.DEVICE)

            self.contract_size = torch.tensor(ApplicationConstants.CONTRACT_SIZE, dtype=torch.float32, device=self.device)
            self.leverage = torch.tensor(ApplicationConstants.LEVERAGE, dtype=torch.float32, device=self.device)
            self.conversion_rate = torch.tensor(c.convert(1, 'EUR', 'GBP'), dtype=torch.float32, device=self.device)
            self.transaction_fee = torch.tensor(ApplicationConstants.TRANSACTION_FEE, dtype=torch.float32, device=self.device)

            # stored on the cpu
            self.is_open = False
    
    def open_trade(self,
                    lot_size: torch.Tensor,
                    open_price: torch.Tensor,
                    trade_type: torch.Tensor,
                    stop_loss: torch.Tensor = None,
                    take_profit: torch.Tensor = None) -> None:
        """
        Open a trade
        """
        self._lot_size = lot_size
        self._open_price = open_price
        self.trade_type = trade_type
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        self.is_open = True

    def calculate_pnl(self, current_price: torch.Tensor) -> torch.Tensor:
        """
        Calculate the profit or loss of the trade
        """
        if not self.is_open:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        if self.trade_type == 1:
             return (current_price - self._open_price) * self._lot_size
        else:
            return (self._open_price - current_price) * self._lot_size
        
    def get_margin(self) -> torch.Tensor:
        """
        Calculate the margin
        """
        if not self.is_open:
             return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        return ((self._lot_size * self.contract_size) / self.leverage) * self.conversion_rate

    def close_trade(self, close_price: torch.Tensor) -> torch.Tensor:
        """
        Close the trade
        """
        if not self.is_open:
             return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        pnl = self.calculate_pnl(close_price)

        self.is_open = False
        # reset the trade
        self._lot_size = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._open_price = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.trade_type = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.stop_loss = None
        self.take_profit = None

        return pnl - self.transaction_fee
    
    def trigger_sl_or_tp(self, current_high: torch.Tensor, current_low: torch.Tensor) -> torch.Tensor:
        """
        This method checks if the current high or low has hit the stop loss or take profit of a trade
        """
        if not self.is_open:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        stop_loss_hit = (self.trade_type == 1 and self.stop_loss is not None and current_low <= self.stop_loss) or \
                (self.trade_type != 1 and self.stop_loss is not None and current_high >= self.stop_loss)
        
        take_profit_hit = (self.trade_type == 1 and self.take_profit is not None and current_high >= self.take_profit) or \
                    (self.trade_type != 1 and self.take_profit is not None and current_low <= self.take_profit)

        if stop_loss_hit:
            return self.close_trade(self.stop_loss)
        elif take_profit_hit:
            return self.close_trade(self.take_profit)

        return torch.tensor(0.0, dtype=torch.float32, device=self.device)

@staticmethod
def check_margin(lot_size: float) -> float:
    """
    Check if the margin is enough to open a trade
    """
    return ((lot_size * ApplicationConstants.CONTRACT_SIZE) / ApplicationConstants.LEVERAGE) * c.convert(1, 'EUR', 'GBP')


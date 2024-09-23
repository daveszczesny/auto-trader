
from datetime import datetime
from typing import Optional, List, Dict, Any
from trade import Trade
from utils.constants import Decision, TradeDecision, TradeType

STARTING_ACCOUNT_BALANCE: float = 1_000
debug_mode: bool = False


class Agent:
    _account_balance: float = 0.0

    def __init__(self, debug_mode: bool = False) -> None:
        self.account_balance = STARTING_ACCOUNT_BALANCE
        self.open_trades: List[Trade] = []

        self.decision = Optional[Decision]
        self.trade_decision = Optional[TradeDecision]
        self.lot_size = 0.01
        self.debug_mode = debug_mode


        self._logs: List[str] = []

        if self.debug_mode:
            self.biggest_profit: float = 0.0
            self.biggest_loss: float = 0.0
            self.trades_executed: int = 0
            self.starting_account_balance: float = self.account_balance
            self.highest_account_balance: float = 0.0
            self.lowest_account_balance: float = self.account_balance
            self.all_trades: List[Dict[str, Any]] = []
            self.wins: int = 0
            self.losses: int = 0
            self.long_wins: int = 0
            self.short_wins: int = 0
            self.long_trades: int = 0
            self.short_trades: int = 0


    @property
    def account_balance(self) -> float:
        return round(self._account_balance, 2)
    
    @account_balance.setter
    def account_balance(self, value: float) -> None:
        self._account_balance = value

    def make_decision(self) -> None:
        """
        Make a decision
        """
        self.think()
        self.lot_size = self.calculate_lot_size()
        self._execute_decision()
    

    """ Override this method """
    def think(self) -> None:
        """
        Think about the next move
        """
        pass

    """ Override this method """
    def calculate_lot_size(self) -> float:
        """
        Calculate the lot size
        """
        return 0.01


    def _execute_decision(self) -> None:
        """
        Execute the decision
        """
        if self.decision == Decision.ENTER_TRADE:
            self._enter_trade()
        elif self.decision == Decision.CLOSE_TRADE:
            self._close_trade()
        else:
            # Do nothing
            pass

    def _enter_trade(self) -> None:
        """
        Enter a trade
        """
        if self.trade_decision == TradeDecision.LONG:
            self._open_long_trade()
        else:
            self._open_short_trade()

    def _open_long_trade(self) -> None:
        """
        Open a long trade
        """
        trade = Trade(self)
        if trade.long(self.lot_size):
            self.open_trades.append(trade)

            if self.debug_mode:
                self.trades_executed += 1
                self.long_trades += 1

                td = {
                    'id': trade.uuid,
                    'count': self.trades_executed,
                    'type': TradeType.LONG,
                    'lot_size': self.lot_size,
                }
                self.all_trades.append(td)

            return
        
        self._logs.append("Failed to open long trade")


    def _open_short_trade(self) -> None:
        """
        Open a short trade
        """
        trade = Trade(self)
        if trade.short(self.lot_size):
            self.open_trades.append(trade)

            if self.debug_mode:
                self.trades_executed += 1
                self.short_trades += 1

                td = {
                    'id': trade.uuid,
                    'count': self.trades_executed,
                    'type': TradeType.SHORT,
                    'lot_size': self.lot_size,
                }
                self.all_trades.append(td)

            return

        self._logs.append("Failed to open short trade")


    def _close_trade(self, trade: Optional[Trade] = None) -> None:
        """
        Close a trade
        """

        if not trade:
            return

        value: float = trade.close_trade()
        self.account_balance += value

        if self.debug_mode:
            self.biggest_profit = max(self.biggest_profit, value)
            self.biggest_loss = min(self.biggest_loss, value)
            
            if value > 0:
                self.wins += 1
                if trade.type_of_trade == TradeType.LONG:
                    self.long_wins += 1
                else:
                    self.short_wins += 1

            else:
                self.losses += 1

            for t in self.all_trades:
                if t['id'] == trade.uuid:
                    t['profit'] = value
                    t['date'] = datetime.now().strftime('%Y_%m_%d_%H_%M')
                    t['account_balance'] = self.account_balance


    def _write_logs(self) -> None:
        """
        Write logs to a file
        """
        filename = f'logs_{datetime.now().strftime("%Y_%m_%d_%H_%M")}.txt'
        with open(filename, 'w') as f:
            for log in self._logs:
                f.write(log + '\n')

            if not self.debug_mode:
                return
        
            avg_win: float = sum(self.wins) / len(self.wins) if self.wins else 0
            avg_loss: float = sum(self.losses) / len(self.losses) if self.losses else 0
            win_rate: float = round(self.wins / self.trades_executed, 2) if self.trades_executed else 0
            long_win_rate: float = round(self.long_wins / self.long_trades, 2) if self.long_trades else 0
            short_win_rate: float = round(self.short_wins / self.short_trades, 2) if self.short_trades else 0

            f.write(f"Starting account balance: {self.starting_account_balance}\n")
            f.write(f"Final account balance: {self.account_balance}\n")
            f.write(f"Profit made: {self.account_balance - self.starting_account_balance}\n")
            f.write(f"Highest account balance: {self.highest_account_balance}\n")
            f.write(f"Lowest account balance: {self.lowest_account_balance}\n")
            f.write(f"Trades executed: {self.trades_executed}\n")
            f.write(f"Wins: {self.wins}\n")
            f.write(f"Losses: {self.losses}\n")
            f.write(f"Win rate: {win_rate}\n")
            f.write(f"Long trades: {self.long_trades}\n")
            f.write(f"Long win rate: {long_win_rate}\n")
            f.write(f"Short trades: {self.short_trades}\n")
            f.write(f"Short win rate: {short_win_rate}\n")
            f.write(f"Biggest profit: {self.biggest_profit}\n")
            f.write(f"Biggest loss: {self.biggest_loss}\n")
            f.write(f"Average winning trade: {avg_win}\n")
            f.write(f"Average losing trade: {avg_loss}\n")

            f.write("\n\n\n")
            for trade in self.all_trades:
                f.write(f"Trade: {trade['type']}\n")
                f.write(f"Profit: {trade['profit']}\n")
                f.write(f"Date: {trade['date']}\n")
                f.write(f"Lot size: {trade['lot_size']}\n")

        trade_list_filename = f"all_trades_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt" 
        with open(trade_list_filename, 'w') as f:
            for trade in self.all_trades:
                f.write(f"{trade['count']}, {trade['account_balance']}")

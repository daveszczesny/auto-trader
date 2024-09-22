import random
import pandas as pd

from agent import Agent
from trade import Trade
from utils.constants import Decision, CloseDecision

class RandomBot:

    logs: list[str] = []

    # agent.account_balance = 1_000.0
    agent: Agent = Agent()

    # Max concurrent trades
    max_concurrent_trades: int = 5

    current_dta: pd.Series | None = None
    decision: Decision | None = None
    close_trade_decision: CloseDecision | None = None
    trades: list[Trade] = []
    
    biggest_loss: float = 0.0
    biggest_profit: float = 0.0
    trades_executed: int = 0
    account_balance_over_time: list[dict] = []

    def make_decision(self):
        """
        Make a decision
        """

        if len(self.trades) >= self.max_concurrent_trades:
            self.decision = Decision.DO_NOTHING
            self.choose_to_close()
            return
        
        self.decision = self.choose_direction()

        lots = self.lots()
        if lots == -1:
            return

        trade = Trade(self.agent)
        if self.decision == Decision.LONG:
            if trade.long(lots):
                self.logs.append(f"Agent is going long, lots: {lots}")
                self.trades_executed += 1
                self.trades.append(trade)
            else:
                self.logs.append("Could not go long")
        elif self.decision == Decision.SHORT:
            if trade.short(lots):
                self.logs.append(f"Agent is going short, lots: {lots}")
                self.trades_executed += 1
                self.trades.append(trade)
            else:
                self.logs.append("Could not go short")

        self.choose_to_close()


    def choose_to_close(self):
        if self.trades:
            self.close_trade_decision = random.choices(
                population=[CloseDecision.CLOSE, CloseDecision.DO_NOTHING],
                weights=[0.08, 0.92]
            )[0]

            if self.close_trade_decision == CloseDecision.CLOSE:
                trade = self.trades.pop(0)
                profit: float = trade.close_trade()

                if profit > self.biggest_profit:
                    self.biggest_profit = profit
                if profit < self.biggest_loss:
                    self.biggest_loss = profit

                self.logs.append(f"Closing trade, profit: {profit}")
                self.agent.account_balance += profit

                self.account_balance_over_time.append(
                    { "trade_no": self.trades_executed, "account_balance": self.agent.account_balance}
                )

    def close_all_trades(self):
        """
        Close all trades
        """
        for trade in self.trades:
            profit: float = trade.close_trade()
            self.agent.account_balance += profit
            self.logs.append(f"Closing trade, profit: {profit}")
            self.account_balance_over_time.append(
                { 
                    "trade_no": self.trades_executed, 
                    "account_balance": str(self.agent.account_balance)
                }
            )

    
    def write_logs(self):

        with open('logs.txt', 'w') as f:
            for log in self.logs:
                f.write(log + '\n')

            f.write(f"\n\n{self.account_balance_over_time}\n")

            f.write(f"\nFinal account balance: {self.agent.account_balance}\n")
            f.write(f"Trades executed: {self.trades_executed}\n")
            f.write(f"Biggest profit: {self.biggest_profit}\n")
            f.write(f"Biggest loss: {self.biggest_loss}\n")

    def data(self, data: pd.Series) -> None:
        """
        Get the data from environment
        """
        self.current_data = data


    def choose_direction(self):
        """
        Make a random decision
        """
        return random.choices(
            population=[Decision.LONG, Decision.SHORT, Decision.DO_NOTHING],
            weights=[0.001, 0.001, 0.998]
        )[0]
    

    def lots(self, _lots: float = 1) -> float:
        """
        Determine lots to trade based on account balance and risk management
        """

        if self.decision == Decision.DO_NOTHING:
            return -1

        # Ensure minimum lot size
        if _lots < 0.01:
            return 0.01

        trade: Trade = Trade(self.agent)
        trade.lots = _lots
        cost_of_lots = trade.get_cost_to_enter_trade()

        # Calculate the ratio of the cost of lots to the account balance
        lot_cost_ratio_to_balance = cost_of_lots / (self.agent.account_balance * 0.4)

        # Determine lots based on risk management principles
        # if lot_cost_ratio_to_balance < 0.01:
        #     lots_used = random.uniform(1, 2)
        # elif lot_cost_ratio_to_balance < 0.05:
        #     lots_used = random.uniform(0.5, 1)
        # elif lot_cost_ratio_to_balance < 0.1:
        #     lots_used = random.uniform(0.1, 0.5)
        # elif lot_cost_ratio_to_balance < 0.2:
        #     lots_used = random.uniform(0.05, 0.1)
        # else:
        #     lots_used = random.uniform(0.01, 0.05)

        return round(0.1, 2)


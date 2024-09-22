class Agent:
    
    _account_balance: float = 0.0

    def __init__(self):
        self.account_balance = 100_000.0


    @property
    def account_balance(self):
        return round(self._account_balance, 2)
    
    @account_balance.setter
    def account_balance(self, value):
        # assert value >= 0, "Account balance cannot be negative"
        self._account_balance = value
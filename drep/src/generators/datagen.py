import numpy as np
import pandas as pd

"""
This module will be used to generate data for the environment
    using different math functions, in order to create a noiseless environment,
    for the AI to train on.
"""

class DataGenerator:
    
    def __init__(self, n_steps: int, base_price: float, noise: float = 0.0, min_price: float = 0.1):
        self.n_steps = n_steps
        self.base_price = base_price
        self.noise = noise
        self.min_price = min_price

    def generate_data(self):
        time = np.arange(self.n_steps)
        bid_close = self.base_price + 0.0001 * np.sin(time) + np.random.normal(0, self.noise, self.n_steps)
        bid_close = np.maximum(bid_close, self.min_price)
        bid_high = bid_close + np.random.uniform(0.01, 0.05, self.n_steps)
        bid_low = bid_close - np.random.uniform(0.01, 0.05, self.n_steps)
        bid_low = np.maximum(bid_low, self.min_price)
        
        data = {
            'bid_close': bid_close,
            'bid_high': bid_high,
            'bid_low': bid_low
        }
        
        df = pd.DataFrame(data)
        df = df.round(5)  # Round to 5 decimal points
        return df

    def save_to_csv(self, df, filename: str):
        df.to_csv(filename, index=False)


# Example usage
dg = DataGenerator(n_steps=1000, base_price=1.2, noise=0.0, min_price=0.1)
df = dg.generate_data()
print(df.head())

dg.save_to_csv(df, 'data.csv')

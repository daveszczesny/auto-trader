
import pandas as pd


# load file from resources
df = pd.read_csv('resources/training_data.csv')

# drop ask_open,ask_high,ask_low,ask_close
df = df.drop(['ask_open','ask_high','ask_low','ask_close', 'ask_volume_sum','bid_volume_sum', 'bid_open'], axis=1)


# save to resources
df.to_csv('resources/training_data2.csv', index=False)
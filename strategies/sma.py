import pandas as pd

class SMAStrategy:
    def __init__(self, short_period, long_period, name="SMA"):
        self.short_period = short_period
        self.long_period = long_period
        self.name = name

    def get_signal(self, data):
        df = pd.DataFrame(data)
        df['sma_short'] = df['close'].rolling(window=self.short_period).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_period).mean()
        if df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1] and df['sma_short'].iloc[-2] <= df['sma_long'].iloc[-2]:
            return 'buy'
        elif df['sma_short'].iloc[-1] < df['sma_long'].iloc[-1] and df['sma_short'].iloc[-2] >= df['sma_long'].iloc[-2]:
            return 'sell'
        return 'hold'
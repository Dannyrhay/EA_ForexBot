from .base_strategy import BaseStrategy
import pandas as pd

class SMAStrategy(BaseStrategy):
    def __init__(self, name, short_period, long_period):
        super().__init__(name)
        self.short_period = short_period
        self.long_period = long_period
        
    def get_signal(self, data, symbol=None):
        """Generate trading signal based on SMA crossover."""
        df = pd.DataFrame(data)
        df['sma_short'] = df['close'].rolling(window=self.short_period).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_period).mean()
        strength = 0.8  # Confidence score for buy/sell signals
        if (df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1] and
                df['sma_short'].iloc[-2] <= df['sma_long'].iloc[-2]):
            return ('buy', strength)
        elif (df['sma_short'].iloc[-1] < df['sma_long'].iloc[-1] and
              df['sma_short'].iloc[-2] >= df['sma_long'].iloc[-2]):
            return ('sell', strength)
        return ('hold', 0.0)
from .base_strategy import BaseStrategy
import pandas as pd

class SupplyDemandStrategy(BaseStrategy):
    def __init__(self, name, window):
        super().__init__(name)
        self.window = window
        
    def get_signal(self, data, symbol=None):
        """Generate trading signal based on supply and demand zones."""
        df = pd.DataFrame(data)
        df['high_max'] = df['high'].rolling(window=self.window).max()
        df['low_min'] = df['low'].rolling(window=self.window).min()
        current_price = df['close'].iloc[-1]
        last_high = df['high_max'].iloc[-1]
        last_low = df['low_min'].iloc[-1]
        strength = 0.8  # Confidence score for buy/sell signals
        if current_price > last_high:
            return ('buy', strength)
        elif current_price < last_low:
            return ('sell', strength)
        return ('hold', 0.0)
from .base_strategy import BaseStrategy
import pandas as pd

class FibonacciStrategy(BaseStrategy):
    def __init__(self, name, levels):
        super().__init__(name)
        self.levels = levels
        
    def get_signal(self, data, symbol=None):
        """Generate trading signal based on Fibonacci retracement levels."""
        df = pd.DataFrame(data)
        swing_high = df['high'].rolling(window=20).max().iloc[-1]
        swing_low = df['low'].rolling(window=20).min().iloc[-1]
        current_price = df['close'].iloc[-1]
        fib_range = swing_high - swing_low
        strength = 0.8  # Confidence score for buy/sell signals
        for level in self.levels:
            fib_price = swing_low + fib_range * level
            if abs(current_price - fib_price) < fib_range * 0.01:
                return ('buy' if current_price > swing_low else 'sell', strength)
        return ('hold', 0.0)
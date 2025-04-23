import pandas as pd

class FibonacciStrategy:
    def __init__(self, levels, name="Fibonacci"):
        self.levels = levels
        self.name = name

    def get_signal(self, data):
        df = pd.DataFrame(data)
        # Simplified Fibonacci: Price near retracement levels
        swing_high = df['high'].rolling(window=20).max().iloc[-1]
        swing_low = df['low'].rolling(window=20).min().iloc[-1]
        current_price = df['close'].iloc[-1]
        fib_range = swing_high - swing_low
        for level in self.levels:
            fib_price = swing_low + fib_range * level
            if abs(current_price - fib_price) < fib_range * 0.01:
                return 'buy' if current_price > swing_low else 'sell'
        return 'hold'
import pandas as pd

class LiquiditySweepStrategy:
    def __init__(self, period, name="LiquiditySweep"):
        self.period = period
        self.name = name

    def get_signal(self, data):
        df = pd.DataFrame(data)
        # Detect liquidity sweep: Price moves below/above recent low/high then reverses
        recent_low = df['low'].rolling(window=self.period).min().iloc[-2]
        recent_high = df['high'].rolling(window=self.period).max().iloc[-2]
        current_low = df['low'].iloc[-1]
        current_high = df['high'].iloc[-1]
        current_close = df['close'].iloc[-1]
        if current_low < recent_low and current_close > recent_low:
            return 'buy'
        elif current_high > recent_high and current_close < recent_high:
            return 'sell'
        return 'hold'
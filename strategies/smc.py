import pandas as pd

class SMCStrategy:
    def __init__(self, name="SMC"):
        self.name = name

    def get_signal(self, data):
        df = pd.DataFrame(data)
        # Simplified SMC: Look for higher highs/lower lows
        if df['high'].iloc[-1] > df['high'].iloc[-2] and df['low'].iloc[-1] > df['low'].iloc[-2]:
            return 'buy'
        elif df['high'].iloc[-1] < df['high'].iloc[-2] and df['low'].iloc[-1] < df['low'].iloc[-2]:
            return 'sell'
        return 'hold'
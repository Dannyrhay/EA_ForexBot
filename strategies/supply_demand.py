import pandas as pd

class SupplyDemandStrategy:
    def __init__(self, window, name="SupplyDemand"):
        self.window = window
        self.name = name

    def get_signal(self, data):
        df = pd.DataFrame(data)
        df['high_max'] = df['high'].rolling(window=self.window).max()
        df['low_min'] = df['low'].rolling(window=self.window).min()
        current_price = df['close'].iloc[-1]
        last_high = df['high_max'].iloc[-1]
        last_low = df['low_min'].iloc[-1]
        if current_price > last_high:
            return 'buy'
        elif current_price < last_low:
            return 'sell'
        return 'hold'
import pandas as pd

class FVGStrategy:
    def __init__(self, gap_threshold, name="FVG"):
        self.gap_threshold = gap_threshold
        self.name = name

    def get_signal(self, data):
        df = pd.DataFrame(data)
        # Fair Value Gap: Detect gaps between candles
        prev_high = df['high'].iloc[-2]
        prev_low = df['low'].iloc[-2]
        curr_high = df['high'].iloc[-1]
        curr_low = df['low'].iloc[-1]
        if curr_low > prev_high + self.gap_threshold:
            return 'buy'
        elif curr_high < prev_low - self.gap_threshold:
            return 'sell'
        return 'hold'
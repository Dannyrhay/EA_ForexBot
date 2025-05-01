from .base_strategy import BaseStrategy
import pandas as pd

class FVGStrategy(BaseStrategy):
    def __init__(self, name, gap_threshold):
        super().__init__(name)
        self.gap_threshold = gap_threshold
        
    def get_signal(self, data, symbol=None):
        """Generate trading signal based on Fair Value Gaps."""
        df = pd.DataFrame(data)
        prev_candle = df.iloc[-3]
        current_candle = df.iloc[-1]
        strength = 0.8  # Confidence score for buy/sell signals
        # Bullish FVG: Low of current candle is higher than high of previous candle
        if current_candle['low'] > prev_candle['high']:
            gap_size = current_candle['low'] - prev_candle['high']
            if gap_size >= self.gap_threshold * prev_candle['close']:
                return ('buy', strength)
        # Bearish FVG: High of current candle is lower than low of previous candle
        elif current_candle['high'] < prev_candle['low']:
            gap_size = prev_candle['low'] - current_candle['high']
            if gap_size >= self.gap_threshold * prev_candle['close']:
                return ('sell', strength)
        return ('hold', 0.0)
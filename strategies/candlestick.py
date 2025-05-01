from .base_strategy import BaseStrategy
import pandas as pd

class CandlestickStrategy(BaseStrategy):
    def __init__(self, name):
        super().__init__(name)
        
    def get_signal(self, data, symbol=None):
        """Generate trading signal based on candlestick patterns."""
        df = pd.DataFrame(data)
        candle1 = df.iloc[-2]
        candle2 = df.iloc[-1]
        strength = 0.8  # Confidence score for buy/sell signals
        if self.is_bullish_engulfing(candle1, candle2):
            return ('buy', strength)
        if self.is_bearish_engulfing(candle1, candle2):
            return ('sell', strength)
        return ('hold', 0.0)

    def is_bullish_engulfing(self, candle1, candle2):
        """Check for bullish engulfing pattern."""
        try:
            if (candle1['close'] < candle1['open'] and
                candle2['close'] > candle2['open'] and
                candle2['close'] > candle1['open'] and
                candle2['open'] < candle1['close']):
                return True
            return False
        except Exception as e:
            return False

    def is_bearish_engulfing(self, candle1, candle2):
        """Check for bearish engulfing pattern."""
        try:
            if (candle1['close'] > candle1['open'] and
                candle2['close'] < candle2['open'] and
                candle2['close'] < candle1['open'] and
                candle2['open'] > candle1['close']):
                return True
            return False
        except Exception as e:
            return False
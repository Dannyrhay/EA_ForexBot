import pandas as pd

class CandlestickStrategy:
    def __init__(self, name="Candlestick"):
        self.name = name

    def get_signal(self, data):
        df = pd.DataFrame(data)
        candle1 = df.iloc[-2]
        candle2 = df.iloc[-1]
        if self.is_bullish_engulfing(candle1, candle2):
            return 'buy'
        elif self.is_bearish_engulfing(candle1, candle2):
            return 'sell'
        return 'hold'

    def is_bullish_engulfing(self, candle1, candle2):
        """Check for bullish engulfing pattern."""
        try:
            if candle1['close'] < candle1['open'] and \
               candle2['close'] > candle2['open'] and \
               candle2['close'] > candle1['open'] and \
               candle2['open'] < candle1['close']:
                return True
            return False
        except Exception as e:
            return False

    def is_bearish_engulfing(self, candle1, candle2):
        """Check for bearish engulfing pattern."""
        try:
            if candle1['close'] > candle1['open'] and \
               candle2['close'] < candle2['open'] and \
               candle2['close'] < candle1['open'] and \
               candle2['open'] > candle1['close']:
                return True
            return False
        except Exception as e:
            return False
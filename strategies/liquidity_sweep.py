import logging
from .base_strategy import BaseStrategy
import pandas as pd

class LiquiditySweepStrategy(BaseStrategy):
    def __init__(self, name, period):
        super().__init__(name)
        self.period = period
        
    def get_signal(self, data, symbol=None, timeframe=None):
        """
        Generate trading signal based on liquidity sweep patterns with trend and volume confirmation.

        Parameters:
        - data: DataFrame with OHLCV data
        - symbol: Trading symbol (optional)
        - timeframe: Timeframe of the data (optional)

        Returns:
        - Tuple[str, float]: (signal, strength) where signal is 'buy', 'sell', or 'hold',
                            and strength is a float between 0.0 and 1.0
        """
        try:
            df = pd.DataFrame(data)
            if not all(col in df for col in ['open', 'close', 'high', 'low']):
                logging.debug(f"Missing required columns for {symbol} on {timeframe}")
                return ('hold', 0.0)

            # Calculate 50-period SMA for trend
            df['sma'] = df['close'].rolling(window=50).mean()
            recent_low = df['low'].rolling(window=self.period).min().iloc[-2]
            recent_high = df['high'].rolling(window=self.period).max().iloc[-2]
            current_low = df['low'].iloc[-1]
            current_high = df['high'].iloc[-1]
            current_close = df['close'].iloc[-1]
            current_sma = df['sma'].iloc[-1] if not pd.isna(df['sma'].iloc[-1]) else current_close
            current_volume = df['tick_volume'].iloc[-1] if 'tick_volume' in df else 0
            avg_volume = df['tick_volume'].rolling(window=20).mean().iloc[-1] if 'tick_volume' in df else 0
            strength = 0.8

            if (current_close > current_sma and
                current_low < recent_low and
                current_close > recent_low and
                current_volume > avg_volume * 1.5):
                logging.debug(f"Buy signal for {symbol} on {timeframe}: Sweep below {recent_low}, closed above, volume high")
                return ('buy', strength)
            elif (current_close < current_sma and
                  current_high > recent_high and
                  current_close < recent_high and
                  current_volume > avg_volume * 1.5):
                logging.debug(f"Sell signal for {symbol} on {timeframe}: Sweep above {recent_high}, closed below, volume high")
                return ('sell', strength)
            else:
                logging.debug(f"No signal for {symbol} on {timeframe}: Trend or volume conditions not met")
                return ('hold', 0.0)
        except Exception as e:
            logging.error(f"Error generating signal for {symbol} on {timeframe}: {str(e)}")
            return ('hold', 0.0)
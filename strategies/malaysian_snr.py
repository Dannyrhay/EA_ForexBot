import pandas as pd
import logging
import numpy as np
from .base_strategy import BaseStrategy

class MalaysianSnRStrategy(BaseStrategy):
    def __init__(self, name, window=20, freshness_window=5, threshold=0.01):
        super().__init__(name)
        self.window = window
        self.freshness_window = freshness_window
        self.freshness_threshold = threshold

    def calculate_rsi(self, data, window=14):
        """
        Calculate the Relative Strength Index (RSI) for the given data.
        
        Parameters:
        - data: DataFrame with OHLCV data
        - window: Period for RSI calculation (default 14)
        
        Returns:
        - float: RSI value, defaults to 50 if calculation fails
        """
        try:
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs.iloc[-1]))
            return 50 if pd.isna(rsi) or np.isinf(rsi) else rsi
        except Exception as e:
            logging.warning(f"Error calculating RSI: {str(e)}")
            return 50

    def is_fresh(self, level_time, level, df, is_support):
        """
        Check if a support/resistance level is fresh (not mitigated by a close beyond it).
        
        Parameters:
        - level_time: Timestamp of the level
        - level: Price level to check
        - df: DataFrame with OHLCV data and 'time' column
        - is_support: Boolean indicating if the level is support (True) or resistance (False)
        
        Returns:
        - bool: True if the level is fresh, False otherwise
        """
        try:
            level_idx = df[df['time'] == pd.to_datetime(level_time)].index[0]
            end_idx = min(level_idx + self.freshness_window, len(df) - 1)
            recent_data = df.iloc[level_idx + 1:end_idx + 1]
            if is_support:
                return not any(recent_data['close'] < level)
            else:
                return not any(recent_data['close'] > level)
        except Exception as e:
            logging.error(f"Error checking level freshness for {level}: {str(e)}")
            return False

    def identify_levels(self, data: pd.DataFrame) -> tuple[list, list]:
        """
        Identify significant support (A) and resistance (V) levels using swing highs/lows.

        Parameters:
        - data: DataFrame with OHLCV data and 'time' column

        Returns:
        - Tuple[List[Tuple[datetime, float]], List[Tuple[datetime, float]]]: 
          (support_levels, resistance_levels) where each level is (time, price)
        """
        try:
            support_levels = []
            resistance_levels = []
            for i in range(5, len(data) - 5):
                if (data['high'].iloc[i] > data['high'].iloc[i-1] and
                    data['high'].iloc[i] > data['high'].iloc[i+1] and
                    data['high'].iloc[i] > data['high'].iloc[i-2] and
                    data['high'].iloc[i] > data['high'].iloc[i+2]):
                    resistance_levels.append((data['time'].iloc[i], data['high'].iloc[i]))
                elif (data['low'].iloc[i] < data['low'].iloc[i-1] and
                      data['low'].iloc[i] < data['low'].iloc[i+1] and
                      data['low'].iloc[i] < data['low'].iloc[i-2] and
                      data['low'].iloc[i] < data['low'].iloc[i+2]):
                    support_levels.append((data['time'].iloc[i], data['low'].iloc[i]))
            return support_levels, resistance_levels
        except Exception as e:
            logging.error(f"Error identifying levels: {str(e)}")
            return ([], [])

    def get_signal(self, data: pd.DataFrame, symbol: str = None, timeframe: str = None) -> tuple[str, float]:
        """
        Generate trading signal based on proximity to fresh support/resistance levels in a trending market,
        enhanced with RSI for overbought/oversold conditions.

        Parameters:
        - data: DataFrame with OHLCV data and 'time' column
        - symbol: Trading symbol (optional)
        - timeframe: Timeframe of the data (optional)

        Returns:
        - Tuple[str, float]: (signal, strength) where signal is 'buy', 'sell', or 'hold',
                            and strength is a float between 0.0 and 1.0
        """
        try:
            support_levels, resistance_levels = self.identify_levels(data)
            if not support_levels or not resistance_levels:
                logging.debug(f"No support or resistance levels found for {symbol} on {timeframe}")
                return ('hold', 0.0)

            # Determine trend based on last two swing highs and lows
            if len(support_levels) >= 2 and len(resistance_levels) >= 2:
                last_sh = support_levels[-1][1]
                prev_sh = support_levels[-2][1]
                last_rh = resistance_levels[-1][1]
                prev_rh = resistance_levels[-2][1]
                if last_sh > prev_sh and last_rh > prev_rh:
                    trend = 'bullish'
                elif last_sh < prev_sh and last_rh < prev_rh:
                    trend = 'bearish'
                else:
                    trend = 'neutral'
            else:
                trend = 'neutral'
                logging.debug(f"Insufficient levels to determine trend for {symbol} on {timeframe}")

            # Calculate RSI
            rsi = self.calculate_rsi(data)
            current_price = data['close'].iloc[-1]
            strength = 0.5

            fresh_supports = [lvl for lvl in support_levels if self.is_fresh(lvl[0], lvl[1], data, True)]
            fresh_resistances = [lvl for lvl in resistance_levels if self.is_fresh(lvl[0], lvl[1], data, False)]

            if fresh_supports and fresh_resistances:
                nearest_support = min(fresh_supports, key=lambda x: abs(x[1] - current_price))
                nearest_resistance = min(fresh_resistances, key=lambda x: abs(x[1] - current_price))
                dist_to_support = abs(current_price - nearest_support[1]) / current_price
                dist_to_resistance = abs(current_price - nearest_resistance[1]) / current_price

                threshold = self.freshness_threshold
                # Adjust signal with RSI
                if trend == 'bullish' and dist_to_support < threshold and rsi < 30:
                    signal = 'buy'
                    strength = 0.8
                    logging.debug(f"Buy signal for {symbol} on {timeframe}: Near support {nearest_support[1]} in bullish trend, RSI={rsi}")
                elif trend == 'bearish' and dist_to_resistance < threshold and rsi > 70:
                    signal = 'sell'
                    strength = 0.8
                    logging.debug(f"Sell signal for {symbol} on {timeframe}: Near resistance {nearest_resistance[1]} in bearish trend, RSI={rsi}")
                elif trend == 'bullish' and dist_to_support < threshold:
                    signal = 'buy'
                    strength = 0.6  # Lower strength without RSI confirmation
                    logging.debug(f"Buy signal for {symbol} on {timeframe}: Near support {nearest_support[1]} in bullish trend (RSI={rsi} not oversold)")
                elif trend == 'bearish' and dist_to_resistance < threshold:
                    signal = 'sell'
                    strength = 0.6  # Lower strength without RSI confirmation
                    logging.debug(f"Sell signal for {symbol} on {timeframe}: Near resistance {nearest_resistance[1]} in bearish trend (RSI={rsi} not overbought)")
                else:
                    signal = 'hold'
                    logging.debug(f"No signal for {symbol} on {timeframe}: Trend={trend}, Dist to support={dist_to_support}, Dist to resistance={dist_to_resistance}, RSI={rsi}")
            else:
                signal = 'hold'
                logging.debug(f"No fresh support or resistance levels for {symbol} on {timeframe}")

            return (signal, strength)
        except Exception as e:
            logging.error(f"Error generating signal for {symbol} on {timeframe}: {str(e)}")
            return ('hold', 0.0)

    def set_config(self, config: dict):
        self.window = config.get('snr_window', self.window)
        self.freshness_window = config.get('snr_freshness_window', self.freshness_window)
        self.freshness_threshold = config.get('snr_threshold', self.freshness_threshold)
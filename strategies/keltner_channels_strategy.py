import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)

class KeltnerChannelsStrategy(BaseStrategy):
    def __init__(self, name, ema_period=20, atr_period=10, atr_multiplier=2.0, strength=0.7):
        super().__init__(name)
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.signal_strength = strength

    def _calculate_atr(self, high, low, close, period):
        # Helper function to calculate Average True Range (ATR)
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Use rolling mean for ATR as per typical Keltner Channel calculation
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr

    def get_indicator_values(self, data: pd.DataFrame):
        # Calculates the Keltner Channel bands
        try:
            df = pd.DataFrame(data)
            if not all(col in df for col in ['high', 'low', 'close']):
                logger.warning(f"KeltnerChannelsStrategy: Missing required columns 'high', 'low', 'close'.")
                return None
            if len(df) < max(self.ema_period, self.atr_period) + 1: # Ensure enough data for calculations
                logger.warning(f"KeltnerChannelsStrategy: Insufficient data. Need {max(self.ema_period, self.atr_period) + 1} bars, got {len(df)}.")
                return None

            # Middle Line: Exponential Moving Average of the typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            middle_line = typical_price.ewm(span=self.ema_period, adjust=False).mean()

            # Calculate ATR
            atr = self._calculate_atr(df['high'], df['low'], df['close'], self.atr_period)

            # Check if ATR calculation was successful and contains valid numbers
            if atr is None or atr.isna().all() or (atr.iloc[-1] == 0 and len(atr) > 0): # Check last value if series is not all NaN
                nan_series = pd.Series([np.nan] * len(df), index=df.index)
                logger.debug(f"KeltnerChannelsStrategy: ATR contains NaN or zero. Returning NaN series for bands.")
                return {'upper_band': nan_series, 'middle_band': middle_line, 'lower_band': nan_series}

            # Calculate Upper and Lower Bands
            upper_band = middle_line + (atr * self.atr_multiplier)
            lower_band = middle_line - (atr * self.atr_multiplier)

            return {'upper_band': upper_band, 'middle_band': middle_line, 'lower_band': lower_band}
        except Exception as e:
            logger.error(f"KeltnerChannelsStrategy: Error in get_indicator_values: {e}", exc_info=True) # Log full traceback
            nan_series = pd.Series([np.nan] * len(data), index=data.index) # Ensure index matches input data
            # Return middle_line as well, even if bands fail, for potential partial feature extraction
            middle_line_fallback = pd.Series([np.nan] * len(data), index=data.index)
            if 'close' in data: # Attempt to calculate a fallback middle line if possible
                 typical_price_fallback = (data.get('high', data['close']) + data.get('low', data['close']) + data['close']) / 3
                 middle_line_fallback = typical_price_fallback.ewm(span=self.ema_period, adjust=False).mean()

            return {'upper_band': nan_series, 'middle_band': middle_line_fallback, 'lower_band': nan_series}

    def get_signal(self, data, symbol=None, timeframe=None):
        logger.debug(f"KeltnerChannelsStrategy ({symbol} {timeframe}): Calculating signal. EMA: {self.ema_period}, ATR Per: {self.atr_period}, ATR Mult: {self.atr_multiplier}")
        try:
            indicator_values = self.get_indicator_values(data)
            # Ensure indicator_values and its components are valid
            if indicator_values is None or \
               indicator_values.get('upper_band') is None or indicator_values['upper_band'].empty or \
               indicator_values.get('lower_band') is None or indicator_values['lower_band'].empty:
                logger.debug(f"KeltnerChannelsStrategy ({symbol} {timeframe}): Could not get indicator values or returned empty/invalid series.")
                return ('hold', 0.0)

            upper_band = indicator_values['upper_band']
            lower_band = indicator_values['lower_band']
            # middle_band = indicator_values['middle_band'] # Not used in this signal logic but available

            df = pd.DataFrame(data)
            # Ensure 'close' column exists and has enough data
            if 'close' not in df.columns or len(df['close']) < 2:
                logger.warning(f"KeltnerChannelsStrategy ({symbol} {timeframe}): 'close' column missing or insufficient data.")
                return ('hold', 0.0)

            current_close = df['close'].iloc[-1]
            # prev_close = df['close'].iloc[-2] # Previous logic's variable, not used in new logic

            current_upper_band = upper_band.iloc[-1]
            current_lower_band = lower_band.iloc[-1]
            # current_middle_band = middle_band.iloc[-1] # Not used in this signal logic

            logger.debug(f"KeltnerChannelsStrategy ({symbol} {timeframe}): Close={current_close:.4f}, Upper={current_upper_band:.4f}, Lower={current_lower_band:.4f}")

            # Check for NaN values before comparison
            if pd.isna(current_close) or pd.isna(current_upper_band) or pd.isna(current_lower_band):
                logger.debug(f"KeltnerChannelsStrategy ({symbol} {timeframe}): NaN values in bands/close at latest point. Close: {current_close}, Upper: {current_upper_band}, Lower: {current_lower_band}")
                return ('hold', 0.0)

            signal = 'hold'
            strength = 0.0

            # --- IMPLEMENTED SUGGESTED NEW LOGIC ---
            if current_close > current_upper_band:
                signal = 'buy' # Interpreting a close above the upper band as a bullish breakout signal
                strength = self.signal_strength
                logger.info(f"KeltnerChannelsStrategy ({symbol} {timeframe}): BUY Signal (Close > Upper Band). Close={current_close:.4f}, Upper={current_upper_band:.4f}")

            elif current_close < current_lower_band:
                signal = 'sell' # Interpreting a close below the lower band as a bearish breakout signal
                strength = self.signal_strength
                logger.info(f"KeltnerChannelsStrategy ({symbol} {timeframe}): SELL Signal (Close < Lower Band). Close={current_close:.4f}, Lower={current_lower_band:.4f}")

            else:
                logger.debug(f"KeltnerChannelsStrategy ({symbol} {timeframe}): No signal based on close vs bands. Signal: HOLD")

            return (signal, strength)

        except Exception as e:
            logger.error(f"KeltnerChannelsStrategy ({symbol} {timeframe}): Error in get_signal: {e}", exc_info=True) # Log full traceback
            return ('hold', 0.0)

    def set_config(self, config: dict):
        # Allows dynamic configuration updates
        self.ema_period = config.get('keltner_ema_period', self.ema_period)
        self.atr_period = config.get('keltner_atr_period', self.atr_period)
        self.atr_multiplier = config.get('keltner_atr_multiplier', self.atr_multiplier)
        self.signal_strength = config.get('keltner_signal_strength', self.signal_strength)
        logger.debug(f"KeltnerChannelsStrategy config set: EMA={self.ema_period}, ATR Per={self.atr_period}, ATR Mult={self.atr_multiplier}, Strength={self.signal_strength}")

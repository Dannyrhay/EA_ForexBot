import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)

class ADXStrategy(BaseStrategy):
    def __init__(self, name, adx_period=14, di_period=14, adx_threshold=25, strength_factor=0.02):
        super().__init__(name)
        self.adx_period = adx_period
        self.di_period = di_period
        self.adx_threshold = adx_threshold
        self.strength_factor = strength_factor

    def _calculate_atr(self, high, low, close, period):
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr

    def _calculate_di(self, high, low, close, atr, period):
        move_up = high.diff()
        move_down = -low.diff()
        plus_dm = pd.Series(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=high.index)
        minus_dm = pd.Series(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=low.index)
        smooth_plus_dm = plus_dm.ewm(alpha=1/period, adjust=False).mean()
        smooth_minus_dm = minus_dm.ewm(alpha=1/period, adjust=False).mean()
        plus_di = (smooth_plus_dm / atr) * 100
        minus_di = (smooth_minus_dm / atr) * 100
        plus_di = plus_di.fillna(0)
        minus_di = minus_di.fillna(0)
        return plus_di, minus_di

    def _calculate_adx(self, plus_di, minus_di, period):
        dx_denominator = plus_di + minus_di
        dx = np.where(dx_denominator == 0, 0, abs(plus_di - minus_di) / dx_denominator * 100)
        dx_series = pd.Series(dx, index=plus_di.index)
        adx = dx_series.ewm(alpha=1/period, adjust=False).mean()
        return adx

    def get_indicator_values(self, data: pd.DataFrame):
        try:
            df = pd.DataFrame(data)
            if not all(col in df for col in ['high', 'low', 'close']):
                return None
            if len(df) < max(self.adx_period, self.di_period) * 2: # Ensure enough data
                return None

            high = df['high']
            low = df['low']
            close = df['close']

            atr = self._calculate_atr(high, low, close, self.di_period)
            if atr.empty or atr.iloc[-1] == 0 or pd.isna(atr.iloc[-1]):
                nan_series = pd.Series([np.nan] * len(df), index=df.index)
                return {'adx': nan_series, 'plus_di': nan_series, 'minus_di': nan_series}

            plus_di, minus_di = self._calculate_di(high, low, close, atr, self.di_period)
            adx = self._calculate_adx(plus_di, minus_di, self.adx_period)
            return {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}
        except Exception as e:
            logger.error(f"ADXStrategy: Error in get_indicator_values: {e}", exc_info=False)
            nan_series = pd.Series([np.nan] * len(data), index=data.index)
            return {'adx': nan_series, 'plus_di': nan_series, 'minus_di': nan_series}

    def get_signal(self, data, symbol=None, timeframe=None):
        logger.debug(f"ADXStrategy ({symbol} {timeframe}): Calculating signal. ADX Period: {self.adx_period}, DI Period: {self.di_period}, ADX Threshold: {self.adx_threshold}")
        try:
            indicator_values = self.get_indicator_values(data)
            if indicator_values is None or indicator_values['adx'].empty:
                logger.debug(f"ADXStrategy ({symbol} {timeframe}): Could not get indicator values or returned empty series.")
                return ('hold', 0.0)

            adx = indicator_values['adx']
            plus_di = indicator_values['plus_di']
            minus_di = indicator_values['minus_di']

            current_adx = adx.iloc[-1]
            current_plus_di = plus_di.iloc[-1]
            current_minus_di = minus_di.iloc[-1]

            logger.debug(f"ADXStrategy ({symbol} {timeframe}): Values: ADX={current_adx:.2f}, +DI={current_plus_di:.2f}, -DI={current_minus_di:.2f}")

            if pd.isna(current_adx) or pd.isna(current_plus_di) or pd.isna(current_minus_di):
                logger.debug(f"ADXStrategy ({symbol} {timeframe}): NaN values in ADX/DI at latest point.")
                return ('hold', 0.0)

            strength = 0.0
            signal = 'hold'

            if current_adx > self.adx_threshold:
                calculated_strength = min(0.9, (current_adx - self.adx_threshold) * self.strength_factor + 0.5)
                logger.debug(f"ADXStrategy ({symbol} {timeframe}): ADX ({current_adx:.2f}) > Threshold ({self.adx_threshold}). Potential trend.")
                if current_plus_di > current_minus_di:
                    signal = 'buy'
                    strength = calculated_strength
                    logger.info(f"ADXStrategy ({symbol} {timeframe}): BUY signal. ADX={current_adx:.2f}, +DI={current_plus_di:.2f} > -DI={current_minus_di:.2f}. Strength={strength:.2f}")
                elif current_minus_di > current_plus_di:
                    signal = 'sell'
                    strength = calculated_strength
                    logger.info(f"ADXStrategy ({symbol} {timeframe}): SELL signal. ADX={current_adx:.2f}, -DI={current_minus_di:.2f} > +DI={current_plus_di:.2f}. Strength={strength:.2f}")
                else:
                    logger.debug(f"ADXStrategy ({symbol} {timeframe}): ADX > threshold, but DIs not crossed decisively (+DI={current_plus_di:.2f}, -DI={current_minus_di:.2f}). Signal: HOLD")
            else:
                logger.debug(f"ADXStrategy ({symbol} {timeframe}): ADX ({current_adx:.2f}) <= Threshold ({self.adx_threshold}). No strong trend. Signal: HOLD")

            return (signal, strength)

        except Exception as e:
            logger.error(f"ADXStrategy ({symbol} {timeframe}): Error in get_signal: {e}", exc_info=True)
            return ('hold', 0.0)

    def set_config(self, config: dict):
        self.adx_period = config.get('adx_period', self.adx_period)
        self.di_period = config.get('adx_di_period', self.di_period)
        self.adx_threshold = config.get('adx_threshold', self.adx_threshold)
        self.strength_factor = config.get('adx_strength_factor', self.strength_factor)
        logger.debug(f"ADXStrategy config set: ADX Period={self.adx_period}, DI Period={self.di_period}, Threshold={self.adx_threshold}")


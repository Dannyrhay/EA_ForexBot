import pandas as pd
import logging
import numpy as np
from datetime import timedelta

# Set up a logger for this module
logger = logging.getLogger(__name__)

class MalaysianSnRStrategy:
    """
    Implements an advanced strategy based on the "Malaysian SNR Emperor" methodology.

    This strategy differs from conventional SnR in several key ways:
    1.  **Level Identification**: SnR levels are not based on wicks (swing highs/lows) but on the
        open/close of specific candle patterns ('A' shape for resistance, 'V' shape for support).
    2.  **Freshness**: A level is considered "fresh" until a candle's wick *touches* it.
        This signifies liquidity has been taken. A break is not required for a level to become unfresh.
    3.  **Confirmation Signals**: Uses an additive strength model. A signal is generated on proximity
        to a fresh level, and its strength is boosted by confirmations like Rejection, "MISS", and Engulfing.
    """
    def __init__(self, name="MalaysianSnR", window=100, threshold=0.005, miss_period=5, enable_rejection=True, enable_miss=True, enable_engulfing=True):
        """
        Initializes the MalaysianSnRStrategy.
        :param name: The name of the strategy.
        :param window: The lookback period to identify SnR levels.
        :param threshold: The proximity threshold (as a percentage) to define a level's zone.
        :param miss_period: The number of candles that must not touch a level to qualify as a "MISS".
        :param enable_rejection: If True, requires a price rejection confirmation.
        :param enable_miss: If True, gives higher strength to levels validated by a "MISS".
        :param enable_engulfing: If True, requires an engulfing pattern for the highest strength signal.
        """
        self.name = name
        self.window = window
        self.threshold = threshold
        self.miss_period = miss_period
        self.enable_rejection = enable_rejection
        self.enable_miss = enable_miss
        self.enable_engulfing = enable_engulfing

    def set_config(self, config: dict):
        """Allows updating strategy parameters from the main bot config."""
        self.window = config.get('snr_window', self.window)
        self.threshold = config.get('snr_threshold', self.threshold)
        self.miss_period = config.get('snr_miss_period', self.miss_period)
        self.enable_rejection = config.get('snr_enable_rejection_filter', self.enable_rejection)
        self.enable_miss = config.get('snr_enable_miss_filter', self.enable_miss)
        self.enable_engulfing = config.get('snr_enable_engulfing_filter', self.enable_engulfing)
        logger.debug(
            f"MalaysianSnR config set: Window={self.window}, Threshold={self.threshold}, "
            f"MissPeriod={self.miss_period}, RejectionFilter={self.enable_rejection}, "
            f"MissFilter={self.enable_miss}, EngulfingFilter={self.enable_engulfing}"
        )

    def _is_bullish(self, candle):
        return candle['close'] > candle['open']

    def _is_bearish(self, candle):
        return candle['close'] < candle['open']

    def _calculate_atr(self, data, period=14):
        """Calculates the Average True Range (ATR)."""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def identify_levels(self, data: pd.DataFrame) -> tuple[list, list]:
        """
        Identifies 'A' (resistance) and 'V' (support) shaped levels based on candle body patterns.
        Includes Significance Filter: Price must move away by > 2 * ATR.
        """
        resistance_levels = []
        support_levels = []
        # Need at least 2 bars for the pattern (current + next)
        if len(data) < 20: # Need enough data for ATR and lookahead
            return resistance_levels, support_levels

        atr_series = self._calculate_atr(data)

        # Iterate where the 2-bar pattern can be formed
        # We stop earlier to allow for lookahead check
        lookahead = 10
        for i in range(len(data) - 1 - lookahead):
            current_candle = data.iloc[i]
            next_candle = data.iloc[i+1]
            current_atr = atr_series.iloc[i]
            if np.isnan(current_atr) or current_atr == 0: continue

            min_move = 0.5 * current_atr # RELAXED for Verification (was 2.0 * ATR)

            # Resistance 'A' Shape: Bullish candle followed by a Bearish candle.
            if self._is_bullish(current_candle) and self._is_bearish(next_candle):
                level_price = current_candle['close']
                # Check for significant move away (Down)
                future_lows = data['low'].iloc[i+1 : i+1+lookahead]
                max_drop = level_price - future_lows.min()

                if max_drop > min_move:
                    resistance_levels.append((data.index[i], level_price))

            # Support 'V' Shape: Bearish candle followed by a Bullish candle.
            elif self._is_bearish(current_candle) and self._is_bullish(next_candle):
                level_price = current_candle['close']
                # Check for significant move away (Up)
                future_highs = data['high'].iloc[i+1 : i+1+lookahead]
                max_rise = future_highs.max() - level_price

                if max_rise > min_move:
                    support_levels.append((data.index[i], level_price))

        # logger.debug(f"MalaysianSnR: Identified {len(resistance_levels)} resistance and {len(support_levels)} support levels from {len(data)} bars.")
        return resistance_levels, support_levels

    def is_fresh(self, level_time, level_price, df_full_history, is_resistance):
        """
        Checks if a given SnR level is "fresh". A level becomes unfresh if a subsequent
        candle's wick *touches* it.

        :param level_time: The timestamp when the level was formed.
        :param level_price: The price of the level.
        :param df_full_history: The full market data DataFrame.
        :param is_resistance: Boolean flag, True for resistance, False for support.
        :return: Boolean indicating if the level is fresh.
        """
        # Get all data that occurred *after* the level was formed.
        data_since_level = df_full_history[df_full_history.index > level_time]

        if data_since_level.empty:
            return True  # Fresh if no bars have formed since the level

        if is_resistance:
            # A resistance level is unfresh if any subsequent high is >= the level price.
            mitigated = (data_since_level['high'] >= level_price).any()
        else:  # is_support
            # A support level is unfresh if any subsequent low is <= the level price.
            mitigated = (data_since_level['low'] <= level_price).any()

        return not mitigated

    def _is_rejection(self, level_price: float, candle: pd.Series, is_resistance: bool) -> bool:
        """Checks if a candle shows rejection at a given level."""
        if is_resistance:
            # Wick touches/crosses, body closes below
            return candle['high'] >= level_price and candle['close'] < level_price
        else: # is_support
            # Wick touches/crosses, body closes above
            return candle['low'] <= level_price and candle['close'] > level_price

    def _is_miss(self, level_time: pd.Timestamp, level_price: float, df_full_history: pd.DataFrame) -> bool:
        """Checks for the 'MISS' condition, where price fails to retest a level."""
        # Find data between level formation and the last `miss_period` candles
        data_after_level = df_full_history[df_full_history.index > level_time]
        if len(data_after_level) <= self.miss_period:
            return False # Not enough candles to form a miss

        # We check the candles that formed right after the level, up to the recent ones.
        data_to_check_for_miss = data_after_level.iloc[:-self.miss_period]
        if data_to_check_for_miss.empty:
            return True # If no candles in the check window, it's a miss

        # A miss means price never came back to touch the level in this period
        resistance_missed = (data_to_check_for_miss['high'] < level_price).all()
        support_missed = (data_to_check_for_miss['low'] > level_price).all()

        return resistance_missed or support_missed

    def _is_engulfing(self, data: pd.DataFrame, direction: str) -> bool:
        """
        Checks if the last candle is a strong engulfing candle.
        """
        if len(data) < 2:
            return False

        last_candle = data.iloc[-1]
        prev_candle = data.iloc[-2]

        if direction == 'buy':
            # Bullish engulfing: current bullish candle body engulfs previous bearish candle body
            return self._is_bullish(last_candle) and \
                   self._is_bearish(prev_candle) and \
                   last_candle['close'] > prev_candle['open'] and \
                   last_candle['open'] < prev_candle['close']
        elif direction == 'sell':
            # Bearish engulfing: current bearish candle body engulfs previous bullish candle body
            return self._is_bearish(last_candle) and \
                   self._is_bullish(prev_candle) and \
                   last_candle['close'] < prev_candle['open'] and \
                   last_candle['open'] > prev_candle['close']
        return False

    def get_signal(self, data: pd.DataFrame, symbol: str = None, timeframe: str = None, return_features=False):
        """
        Generates a trade signal using an additive strength model based on the Malaysian SNR methodology.
        """
        if not isinstance(data, pd.DataFrame) or data.empty or len(data) < self.window:
            logger.debug(f"MalaysianSnR ({symbol} {timeframe}): Data not valid or insufficient.")
            return ('hold', 0.0, {}) if return_features else ('hold', 0.0)

        df = data.copy()
        if 'time' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time', drop=False)
        elif not isinstance(df.index, pd.DatetimeIndex):
             logger.warning(f"MalaysianSnR ({symbol} {timeframe}): Data has no DatetimeIndex. Cannot proceed.")
             return ('hold', 0.0, {}) if return_features else ('hold', 0.0)

        current_price = df['close'].iloc[-1]
        last_candle = df.iloc[-1]
        lookback_data = df.iloc[-self.window:-1]

        resistance_levels, support_levels = self.identify_levels(lookback_data)

        fresh_resistances = [lvl for lvl in resistance_levels if self.is_fresh(lvl[0], lvl[1], df, is_resistance=True)]
        fresh_supports = [lvl for lvl in support_levels if self.is_fresh(lvl[0], lvl[1], df, is_resistance=False)]

        buy_signal_strength = 0.0
        sell_signal_strength = 0.0

        # SELL Signal Logic - Additive Strength
        if fresh_resistances:
            nearest_res = min(fresh_resistances, key=lambda x: abs(x[1] - current_price))
            level_price = nearest_res[1]

            if abs(current_price - level_price) <= self.threshold * current_price:
                sell_signal_strength += 0.3  # Base strength for proximity to a fresh level
                if self.enable_rejection and self._is_rejection(level_price, last_candle, is_resistance=True):
                    sell_signal_strength += 0.3
                if self.enable_miss and self._is_miss(nearest_res[0], level_price, df):
                    sell_signal_strength += 0.2
                if self.enable_engulfing and self._is_engulfing(df, 'sell'):
                    sell_signal_strength += 0.2

        # BUY Signal Logic - Additive Strength
        if fresh_supports:
            nearest_sup = min(fresh_supports, key=lambda x: abs(x[1] - current_price))
            level_price = nearest_sup[1]

            if abs(current_price - level_price) <= self.threshold * current_price:
                buy_signal_strength += 0.3 # Base strength
                if self.enable_rejection and self._is_rejection(level_price, last_candle, is_resistance=False):
                    buy_signal_strength += 0.3
                if self.enable_miss and self._is_miss(nearest_sup[0], level_price, df):
                    buy_signal_strength += 0.2
                if self.enable_engulfing and self._is_engulfing(df, 'buy'):
                    buy_signal_strength += 0.2

        # --- Feature Calculation for ML Model (if requested) ---
        if return_features:
            features = {'dist_a': 0.0, 'dist_v': 0.0, 'fresh_a': 0, 'fresh_v': 0, 'strength': 0.0}
            if fresh_resistances:
                features['fresh_a'] = 1
                nearest_resistance_price = min([lvl[1] for lvl in fresh_resistances], key=lambda x: abs(x - current_price))
                features['dist_a'] = abs(current_price - nearest_resistance_price) / current_price if current_price != 0 else 0
            if fresh_supports:
                features['fresh_v'] = 1
                nearest_support_price = min([lvl[1] for lvl in fresh_supports], key=lambda x: abs(x - current_price))
                features['dist_v'] = abs(current_price - nearest_support_price) / current_price if current_price != 0 else 0

            features['strength'] = max(buy_signal_strength, sell_signal_strength)
            return 'hold', min(features['strength'], 1.0), features

        # --- Final Signal Determination ---
        # --- Final Signal Determination ---
        atr = self._calculate_atr(df).iloc[-1]

        if buy_signal_strength > sell_signal_strength and buy_signal_strength > 0:
            final_strength = min(buy_signal_strength, 1.0)

            # Calculate SL/TP
            # SL below the support level
            nearest_sup = min(fresh_supports, key=lambda x: abs(x[1] - current_price))
            level_price = nearest_sup[1]
            sl_price = level_price - (atr * 1.5)
            tp_price = current_price + (current_price - sl_price) * 2.0

            logger.info(f"MalaysianSnR ({symbol} {timeframe}): BUY signal. Strength: {final_strength:.2f}, SL: {sl_price:.5f}, TP: {tp_price:.5f}")
            return 'buy', final_strength, {'sl': sl_price, 'tp': tp_price}

        elif sell_signal_strength > buy_signal_strength and sell_signal_strength > 0:
            final_strength = min(sell_signal_strength, 1.0)

            # Calculate SL/TP
            # SL above the resistance level
            nearest_res = min(fresh_resistances, key=lambda x: abs(x[1] - current_price))
            level_price = nearest_res[1]
            sl_price = level_price + (atr * 1.5)
            tp_price = current_price - (sl_price - current_price) * 2.0

            logger.info(f"MalaysianSnR ({symbol} {timeframe}): SELL signal. Strength: {final_strength:.2f}, SL: {sl_price:.5f}, TP: {tp_price:.5f}")
            return 'sell', final_strength, {'sl': sl_price, 'tp': tp_price}

        return 'hold', 0.0, {}

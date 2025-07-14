import logging
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

# Configure logging for the strategy
logger = logging.getLogger(__name__)

class LiquiditySweepStrategy(BaseStrategy):

    def __init__(self, name, **params):
        super().__init__(name)
        # Set default parameters and override with any provided in the config
        self.params = {
            'lookback_period': 20,
            'eq_level_tolerance': 0.0005, # 0.05% tolerance for price equality
            'ob_retrace_pct': 0.5, # 50% retracement for a valid OB
            'volume_multiplier': 1.5,
            'enable_fvg': True,
            'enable_mss_confirmation': True,
            **params
        }
        logger.info(f"{self.name} strategy initialized with parameters: {self.params}")

    # --- HELPER FUNCTIONS for SMC ANALYSIS ---

    def _find_equal_highs_lows(self, df, lookback):
        """
        Identifies recent equal highs (EQH) and equal lows (EQL) within a lookback period.

        Args:
            df (pd.DataFrame): The OHLCV data.
            lookback (int): The number of recent candles to scan.

        Returns:
            tuple: A tuple containing the price of equal highs and equal lows, or (None, None).
        """
        recent_data = df.iloc[-lookback:-1] # Exclude the most recent candle
        if recent_data.empty:
            return None, None

        # Find potential equal highs
        highs = recent_data['high']
        # Check for clusters of highs within a small tolerance
        for i in range(len(highs)):
            for j in range(i + 1, len(highs)):
                if abs(highs.iloc[i] - highs.iloc[j]) / highs.iloc[i] <= self.params['eq_level_tolerance']:
                    eqh = max(highs.iloc[i], highs.iloc[j])
                    return eqh, None # Return first found for simplicity

        # Find potential equal lows
        lows = recent_data['low']
        # Check for clusters of lows within a small tolerance
        for i in range(len(lows)):
            for j in range(i + 1, len(lows)):
                if abs(lows.iloc[i] - lows.iloc[j]) / lows.iloc[i] <= self.params['eq_level_tolerance']:
                    eql = min(lows.iloc[i], lows.iloc[j])
                    return None, eql # Return first found

        return None, None

    def _find_order_block(self, df, lookback):
        """
        Identifies the most recent valid bullish or bearish order block (OB).
        A bearish OB is the last up-candle before a strong down-move.
        A bullish OB is the last down-candle before a strong up-move.

        Args:
            df (pd.DataFrame): The OHLCV data.
            lookback (int): The number of recent candles to scan.

        Returns:
            tuple: (ob_type, ob_range) where ob_type is 'bullish' or 'bearish'
                   and ob_range is a dict {'high': float, 'low': float}. Returns (None, None) if not found.
        """
        data = df.iloc[-lookback:]
        for i in range(len(data) - 2, 0, -1):
            # Potential Bearish OB (last up-candle)
            if data['close'].iloc[i] > data['open'].iloc[i]:
                move_after = data['low'].iloc[i+1]
                ob_low = data['low'].iloc[i]
                # Check for a strong move down that breaks the OB's low
                if move_after < ob_low:
                    # Check if the move was significant (retraced past the OB's body)
                    if data['close'].iloc[i+1] < data['open'].iloc[i] - (abs(data['open'].iloc[i] - data['close'].iloc[i]) * self.params['ob_retrace_pct']):
                        return 'bearish', {'high': data['high'].iloc[i], 'low': data['low'].iloc[i]}

            # Potential Bullish OB (last down-candle)
            if data['close'].iloc[i] < data['open'].iloc[i]:
                move_after = data['high'].iloc[i+1]
                ob_high = data['high'].iloc[i]
                # Check for a strong move up that breaks the OB's high
                if move_after > ob_high:
                     # Check if the move was significant
                    if data['close'].iloc[i+1] > data['open'].iloc[i] + (abs(data['open'].iloc[i] - data['close'].iloc[i]) * self.params['ob_retrace_pct']):
                        return 'bullish', {'high': data['high'].iloc[i], 'low': data['low'].iloc[i]}

        return None, None

    def _find_fair_value_gap(self, df):
        """
        Identifies the most recent Fair Value Gap (FVG).
        A bullish FVG is a gap between candle 1's high and candle 3's low.
        A bearish FVG is a gap between candle 1's low and candle 3's high.

        Args:
            df (pd.DataFrame): The OHLCV data.

        Returns:
            tuple: (fvg_type, fvg_range) or (None, None).
        """
        if len(df) < 3:
            return None, None

        # Check for Bearish FVG (gap to be filled by upward price movement)
        candle1_low = df['low'].iloc[-3]
        candle3_high = df['high'].iloc[-1]
        if candle1_low > candle3_high:
            return 'bearish', {'high': candle1_low, 'low': candle3_high}

        # Check for Bullish FVG (gap to be filled by downward price movement)
        candle1_high = df['high'].iloc[-3]
        candle3_low = df['low'].iloc[-1]
        if candle1_high < candle3_low:
            return 'bullish', {'high': candle1_high, 'low': candle3_low}

        return None, None

    def _check_market_structure_shift(self, df, lookback, direction):
        """
        Checks for a Market Structure Shift (MSS) or Break of Structure (BOS).
        For a buy signal, looks for a break of a recent minor high.
        For a sell signal, looks for a break of a recent minor low.

        Args:
            df (pd.DataFrame): The OHLCV data.
            lookback (int): The number of recent candles to find a swing point.
            direction (str): 'buy' or 'sell'.

        Returns:
            bool: True if a shift is confirmed, False otherwise.
        """
        if not self.params['enable_mss_confirmation']:
            return True # Skip check if disabled

        recent_data = df.iloc[-lookback:]
        current_price = df['close'].iloc[-1]

        if direction == 'buy':
            # Find the most recent significant swing high to break
            swing_high = recent_data['high'].iloc[:-1].max()
            if current_price > swing_high:
                logger.debug(f"MSS confirmed for BUY signal. Price {current_price} broke swing high {swing_high}.")
                return True
        elif direction == 'sell':
            # Find the most recent significant swing low to break
            swing_low = recent_data['low'].iloc[:-1].min()
            if current_price < swing_low:
                logger.debug(f"MSS confirmed for SELL signal. Price {current_price} broke swing low {swing_low}.")
                return True

        return False

    # --- MAIN SIGNAL GENERATION LOGIC ---

    def get_signal(self, data, symbol=None, timeframe=None):
        """
        Generate a trading signal based on the advanced liquidity sweep strategy.

        Returns:
            tuple: (signal, strength, parameters)
                   - signal (str): 'buy', 'sell', or 'hold'.
                   - strength (float): Signal confidence from 0.0 to 1.0.
                   - parameters (dict): SL/TP parameters for the trade executor.
        """
        try:
            df = data.copy()
            if len(df) < self.params['lookback_period']:
                return 'hold', 0.0, None

            # --- 1. Identify Liquidity and Points of Interest ---
            lookback = self.params['lookback_period']
            eqh, eql = self._find_equal_highs_lows(df, lookback)
            ob_type, ob_range = self._find_order_block(df, lookback)
            fvg_type, fvg_range = self._find_fair_value_gap(df) if self.params['enable_fvg'] else (None, None)

            # --- 2. Define Current Market State ---
            current_candle = df.iloc[-1]
            avg_volume = df['tick_volume'].iloc[-lookback:-1].mean()
            is_volume_spike = current_candle['tick_volume'] > avg_volume * self.params['volume_multiplier']

            # --- 3. Check for BUY Signal (Sweep of Lows) ---
            liquidity_level_swept = eql if eql else df['low'].iloc[-lookback:-1].min()
            if current_candle['low'] < liquidity_level_swept and current_candle['close'] > liquidity_level_swept:
                if is_volume_spike:
                    logger.debug(f"BUY Sweep detected for {symbol} on {timeframe}. Swept low: {liquidity_level_swept:.5f}")
                    strength = 0.7  # Base strength for a sweep
                    confluence = 0

                    # Check for confluence with POIs
                    if ob_type == 'bullish' and current_candle['low'] <= ob_range['high']:
                        confluence += 1
                        logger.debug("Confluence: Tapped into a bullish Order Block.")
                    if fvg_type == 'bullish' and current_candle['low'] <= fvg_range['high']:
                        confluence += 1
                        logger.debug("Confluence: Tapped into a bullish FVG.")

                    # Check for MSS confirmation
                    if self._check_market_structure_shift(df, lookback, 'buy'):
                        strength += 0.1 * confluence
                        strength = min(strength, 1.0) # Cap strength at 1.0

                        # Define SL/TP based on the structure
                        sl = current_candle['low'] - (df['high'].iloc[-lookback:-1].max() - df['low'].iloc[-lookback:-1].min()) * 0.1
                        tp = current_candle['close'] + (current_candle['close'] - sl) * 1.5 # Example 1.5 R:R
                        trade_params = {'sl': sl, 'tp': tp, 'source_strategy': self.name}

                        logger.info(f"BUY SIGNAL for {symbol} on {timeframe}. Strength: {strength:.2f}")
                        return 'buy', strength, trade_params

            # --- 4. Check for SELL Signal (Sweep of Highs) ---
            liquidity_level_swept = eqh if eqh else df['high'].iloc[-lookback:-1].max()
            if current_candle['high'] > liquidity_level_swept and current_candle['close'] < liquidity_level_swept:
                if is_volume_spike:
                    logger.debug(f"SELL Sweep detected for {symbol} on {timeframe}. Swept high: {liquidity_level_swept:.5f}")
                    strength = 0.7 # Base strength
                    confluence = 0

                    # Check for confluence with POIs
                    if ob_type == 'bearish' and current_candle['high'] >= ob_range['low']:
                        confluence += 1
                        logger.debug("Confluence: Tapped into a bearish Order Block.")
                    if fvg_type == 'bearish' and current_candle['high'] >= fvg_range['low']:
                        confluence += 1
                        logger.debug("Confluence: Tapped into a bearish FVG.")

                    # Check for MSS confirmation
                    if self._check_market_structure_shift(df, lookback, 'sell'):
                        strength += 0.1 * confluence
                        strength = min(strength, 1.0)

                        # Define SL/TP
                        sl = current_candle['high'] + (df['high'].iloc[-lookback:-1].max() - df['low'].iloc[-lookback:-1].min()) * 0.1
                        tp = current_candle['close'] - (sl - current_candle['close']) * 1.5
                        trade_params = {'sl': sl, 'tp': tp, 'source_strategy': self.name}

                        logger.info(f"SELL SIGNAL for {symbol} on {timeframe}. Strength: {strength:.2f}")
                        return 'sell', strength, trade_params

            return 'hold', 0.0, None

        except Exception as e:
            logger.error(f"Error in {self.name} for {symbol} on {timeframe}: {e}", exc_info=True)
            return 'hold', 0.0, None


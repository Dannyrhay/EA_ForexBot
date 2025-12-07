from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FibonacciStrategy(BaseStrategy):
    """
    Implements the Fibonacci Golden Zone trading strategy.

    This strategy identifies a dominant trend, waits for a price retracement into the
    "Golden Zone" (between 38.2% and 61.8% Fibonacci levels), and enters a trade
    only after a confirmed rejection of the 61.8% level.

    The strategy provides its own Stop Loss (at the 100% level) and Take Profit
    (at the 0% level) for a self-contained trading setup.
    """
    def __init__(self, name, swing_lookback=100, trend_ema_period=50, strength=0.7):
        super().__init__(name)
        self.swing_lookback = swing_lookback
        self.trend_ema_period = trend_ema_period
        self.strength = strength
        self.fib_levels = {
            'entry': 0.236, # Relaxed entry: Start of the Zone (was 0.382)
            'test': 0.382,  # Relaxed test: Must touch this level (was 0.618)
            'deep': 0.618   # Deepest part of the zone
        }

    def get_signal(self, data: pd.DataFrame, symbol: str = None, timeframe: str = None):
        """
        Generates a trading signal based on the Fibonacci Golden Zone strategy.

        Returns:
            - A tuple of ('hold', 0.0, None) if no valid signal is found.
            - A tuple of (signal, strength, params) where signal is 'buy' or 'sell',
              and params is a dict {'sl': stop_loss_price, 'tp': take_profit_price}.
        """
        # Ensure sufficient data for lookback and EMA calculation
        if not isinstance(data, pd.DataFrame) or data.empty or len(data) < max(self.swing_lookback, self.trend_ema_period):
            return ('hold', 0.0, None)

        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return ('hold', 0.0, None)

        # 1. Determine the Dominant Trend using a long-term EMA
        data['ema_trend'] = data['close'].ewm(span=self.trend_ema_period, adjust=False).mean()
        last_close = data['close'].iloc[-1]
        last_ema = data['ema_trend'].iloc[-1]

        is_uptrend = last_close > last_ema
        is_downtrend = last_close < last_ema

        # Look at the last few candles to make sure the trend is stable
        recent_closes = data['close'].iloc[-3:]
        recent_emas = data['ema_trend'].iloc[-3:]
        if (is_uptrend and not (recent_closes > recent_emas).all()) or \
           (is_downtrend and not (recent_closes < recent_emas).all()):
            return ('hold', 0.0, None) # Trend is not stable, hold.

        try:
            # 2. Identify the most recent significant swing within the lookback period
            lookback_data = data.iloc[-self.swing_lookback:]
            swing_high_price = lookback_data['high'].max()
            swing_low_price = lookback_data['low'].min()

            # Ensure the swing is valid
            if pd.isna(swing_high_price) or pd.isna(swing_low_price) or swing_high_price == swing_low_price:
                return ('hold', 0.0, None)

            # Scenario for a BUY signal (Dominant trend is UP)
            if is_uptrend:
                # In an uptrend, we are looking for a retracement DOWN from a swing high.
                # The swing should be from a low up to a high.
                fib_range = swing_high_price - swing_low_price
                if fib_range <= 0: return ('hold', 0.0, None)

                # Define the Zone levels for a BUY
                # Zone starts at 23.6% retracement, requires test of 38.2%
                zone_start = swing_high_price - (fib_range * self.fib_levels['entry']) # 23.6% level
                test_level = swing_high_price - (fib_range * self.fib_levels['test'])  # 38.2% level
                deep_level = swing_high_price - (fib_range * self.fib_levels['deep'])  # 61.8% level

                # Check for entry conditions on the last fully formed candle
                prev_candle = data.iloc[-2]

                # Condition 1: Price must have entered the Zone (passed 23.6%)
                price_entered_zone = prev_candle['low'] <= zone_start

                # Condition 2: Must have tested the "test" level (38.2%)
                # Relaxed from 61.8% to catch strong trends
                level_tested = prev_candle['low'] <= test_level

                # Condition 3: Confirmation - Candle closed back ABOVE the test level
                # This shows rejection of lower prices.
                closed_above_level = prev_candle['close'] > test_level

                if price_entered_zone and level_tested and closed_above_level:
                    sl_price = swing_low_price
                    # Target -0.27 extension for better R:R
                    tp_price = swing_high_price + (fib_range * 0.27)
                    # [NEW] Use Limit Order at the test level (38.2%) for better entry
                    limit_price = test_level
                    trade_params = {'sl': sl_price, 'tp': tp_price, 'limit_price': limit_price, 'source_strategy': 'Fibonacci'}
                    logger.info(f"Fibonacci BUY for {symbol} on {timeframe}: Rejection confirmed at {test_level:.5f}. Limit: {limit_price:.5f} SL: {sl_price:.5f} TP: {tp_price:.5f}")
                    return ('buy', self.strength, trade_params)
                else:
                    logger.debug(f"Fibonacci BUY failed for {symbol}: Entered={price_entered_zone}, Tested382={level_tested}, ClosedAbove={closed_above_level}, Zone=[{test_level:.5f}-{zone_start:.5f}], PrevLow={prev_candle['low']:.5f}, PrevClose={prev_candle['close']:.5f}")

            # Scenario for a SELL signal (Dominant trend is DOWN)
            elif is_downtrend:
                # In a downtrend, we are looking for a retracement UP from a swing low.
                # The swing should be from a high down to a low.
                fib_range = swing_high_price - swing_low_price
                if fib_range <= 0: return ('hold', 0.0, None)

                # Define the Zone levels for a SELL
                zone_start = swing_low_price + (fib_range * self.fib_levels['entry']) # 23.6% level
                test_level = swing_low_price + (fib_range * self.fib_levels['test'])  # 38.2% level
                deep_level = swing_low_price + (fib_range * self.fib_levels['deep'])  # 61.8% level

                # Check for entry conditions on the last fully formed candle
                prev_candle = data.iloc[-2]

                # Condition 1: Price must have entered the Zone
                price_entered_zone = prev_candle['high'] >= zone_start

                # Condition 2: Must have tested the "test" level
                level_tested = prev_candle['high'] >= test_level

                # Condition 3: Confirmation - Candle closed back BELOW the test level
                closed_below_level = prev_candle['close'] < test_level

                if price_entered_zone and level_tested and closed_below_level:
                    sl_price = swing_high_price
                    # Target -0.27 extension for better R:R
                    tp_price = swing_low_price - (fib_range * 0.27)
                    # [NEW] Use Limit Order at the test level (38.2%) for better entry
                    limit_price = test_level
                    trade_params = {'sl': sl_price, 'tp': tp_price, 'limit_price': limit_price, 'source_strategy': 'Fibonacci'}
                    logger.info(f"Fibonacci SELL for {symbol} on {timeframe}: Rejection confirmed at {test_level:.5f}. Limit: {limit_price:.5f} SL: {sl_price:.5f} TP: {tp_price:.5f}")
                    return ('sell', self.strength, trade_params)
                else:
                    logger.debug(f"Fibonacci SELL failed for {symbol}: Entered={price_entered_zone}, Tested382={level_tested}, ClosedBelow={closed_below_level}, Zone=[{zone_start:.5f}-{test_level:.5f}], PrevHigh={prev_candle['high']:.5f}, PrevClose={prev_candle['close']:.5f}")

            return ('hold', 0.0, None)

        except Exception as e:
            # print(f"Fibonacci: Error in get_signal for {symbol} on {timeframe}: {e}")
            return ('hold', 0.0, None)

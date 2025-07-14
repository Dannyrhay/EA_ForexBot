from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class FibonacciStrategy(BaseStrategy):
    """
    Implements the Fibonacci Golden Zone trading strategy.

    This strategy identifies a dominant trend, waits for a price retracement into the
    "Golden Zone" (between 38.2% and 61.8% Fibonacci levels), and enters a trade
    only after a confirmed rejection of the 61.8% level.

    The strategy provides its own Stop Loss (at the 100% level) and Take Profit
    (at the 0% level) for a self-contained trading setup.
    """
    def __init__(self, name, swing_lookback=200, trend_ema_period=200, strength=0.85):
        super().__init__(name)
        self.swing_lookback = swing_lookback
        self.trend_ema_period = trend_ema_period
        self.strength = strength
        self.fib_levels = {
            'level_1': 0.382, # Start of the Golden Zone
            'level_2': 0.618  # Key rejection level (end of Golden Zone)
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

                # Define the Golden Zone levels for a BUY
                golden_zone_start = swing_high_price - (fib_range * self.fib_levels['level_1']) # 38.2% level
                golden_zone_end = swing_high_price - (fib_range * self.fib_levels['level_2'])   # 61.8% level

                # Check for entry conditions on the last fully formed candle
                prev_candle = data.iloc[-2]

                # Condition 1: Price must have entered the Golden Zone
                # The low of the previous candle went into or below the zone
                price_entered_zone = prev_candle['low'] <= golden_zone_start

                # Condition 2: 61.8% level must have been tested
                # The low of the previous candle touched or went below the 61.8% level
                level_tested = prev_candle['low'] <= golden_zone_end

                # Condition 3: Confirmation - Candle closed back ABOVE the 61.8% level
                # This shows rejection of lower prices.
                closed_above_level = prev_candle['close'] > golden_zone_end

                if price_entered_zone and level_tested and closed_above_level:
                    sl_price = swing_low_price
                    tp_price = swing_high_price
                    trade_params = {'sl': sl_price, 'tp': tp_price, 'source_strategy': 'Fibonacci'}
                    # print(f"Fibonacci BUY for {symbol} on {timeframe}: Rejection confirmed at {golden_zone_end:.5f}. SL: {sl_price:.5f} TP: {tp_price:.5f}")
                    return ('buy', self.strength, trade_params)

            # Scenario for a SELL signal (Dominant trend is DOWN)
            elif is_downtrend:
                # In a downtrend, we are looking for a retracement UP from a swing low.
                # The swing should be from a high down to a low.
                fib_range = swing_high_price - swing_low_price
                if fib_range <= 0: return ('hold', 0.0, None)

                # Define the Golden Zone levels for a SELL
                golden_zone_start = swing_low_price + (fib_range * self.fib_levels['level_1']) # 38.2% level
                golden_zone_end = swing_low_price + (fib_range * self.fib_levels['level_2'])   # 61.8% level

                # Check for entry conditions on the last fully formed candle
                prev_candle = data.iloc[-2]

                # Condition 1: Price must have entered the Golden Zone
                price_entered_zone = prev_candle['high'] >= golden_zone_start

                # Condition 2: 61.8% level must have been tested
                level_tested = prev_candle['high'] >= golden_zone_end

                # Condition 3: Confirmation - Candle closed back BELOW the 61.8% level
                closed_below_level = prev_candle['close'] < golden_zone_end

                if price_entered_zone and level_tested and closed_below_level:
                    sl_price = swing_high_price
                    tp_price = swing_low_price
                    trade_params = {'sl': sl_price, 'tp': tp_price, 'source_strategy': 'Fibonacci'}
                    # print(f"Fibonacci SELL for {symbol} on {timeframe}: Rejection confirmed at {golden_zone_end:.5f}. SL: {sl_price:.5f} TP: {tp_price:.5f}")
                    return ('sell', self.strength, trade_params)

            return ('hold', 0.0, None)

        except Exception as e:
            # print(f"Fibonacci: Error in get_signal for {symbol} on {timeframe}: {e}")
            return ('hold', 0.0, None)

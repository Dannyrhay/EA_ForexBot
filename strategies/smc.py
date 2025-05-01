from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from utils.mt5_connection import get_data
import MetaTrader5 as mt5
from datetime import datetime, timedelta

class SMCStrategy(BaseStrategy):
    def __init__(self, name, higher_timeframe='D1', min_ob_size=0.0005, fvg_threshold=0.0005, liquidity_tolerance=0.005, trade_cooldown=30):
        super().__init__(name)
        self.higher_timeframe = higher_timeframe
        self.min_ob_size = min_ob_size
        self.fvg_threshold = fvg_threshold
        self.liquidity_tolerance = liquidity_tolerance
        self.trade_cooldown = trade_cooldown
        self.last_trade_times = {}
        self.mt5_timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M15': mt5.TIMEFRAME_M15,
            'M45': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }

    def in_cooldown(self, symbol):
        last_trade = self.last_trade_times.get(symbol)
        if last_trade and (datetime.now() - last_trade).total_seconds() < self.trade_cooldown * 60:
            print(f"SMC: Cooldown active for {symbol}")
            return True
        return False

    def identify_market_structure(self, data):
        swings = []
        for i in range(2, len(data) - 2):
            if (data['high'].iloc[i] > data['high'].iloc[i-1] and
                data['high'].iloc[i] > data['high'].iloc[i+1] and
                data['high'].iloc[i] > data['high'].iloc[i-2] and
                data['high'].iloc[i] > data['high'].iloc[i+2]):
                swings.append(('high', i, data['high'].iloc[i]))
            elif (data['low'].iloc[i] < data['low'].iloc[i-1] and
                  data['low'].iloc[i] < data['low'].iloc[i+1] and
                  data['low'].iloc[i] < data['low'].iloc[i-2] and
                  data['low'].iloc[i] < data['low'].iloc[i+2]):
                swings.append(('low', i, data['low'].iloc[i]))

        trend = 'neutral'
        last_high = last_low = None
        for swing_type, idx, price in swings:
            if swing_type == 'high':
                last_high = price
                if last_low and data['high'].iloc[-1] > last_high:
                    trend = 'bullish'
            elif swing_type == 'low':
                last_low = price
                if last_high and data['low'].iloc[-1] < last_low:
                    trend = 'bearish'
        return trend, swings

    def identify_liquidity_zones(self, data):
        equal_highs = []
        equal_lows = []
        window = 10
        for i in range(window, len(data)):
            highs = data['high'].iloc[i-window:i]
            lows = data['low'].iloc[i-window:i]
            current_price = data['close'].iloc[i]
            if max(highs) - min(highs) < self.liquidity_tolerance * current_price:
                equal_highs.append((i, max(highs)))
            if max(lows) - min(lows) < self.liquidity_tolerance * current_price:
                equal_lows.append((i, min(lows)))
            round_level = round(current_price / 0.005) * 0.005
            if abs(current_price - round_level) / current_price < self.liquidity_tolerance:
                equal_highs.append((i, round_level))
                equal_lows.append((i, round_level))
        return equal_highs, equal_lows

    def identify_order_blocks(self, data, trend, swings):
        order_blocks = []
        for i in range(3, len(data) - 1):
            if trend == 'bullish':
                if (data['close'].iloc[i-1] < data['open'].iloc[i-1] and
                    data['close'].iloc[i] > data['open'].iloc[i] and
                    data['close'].iloc[i] > data['open'].iloc[i-1] and
                    data['open'].iloc[i] < data['close'].iloc[i-1] and
                    (data['high'].iloc[i-1] - data['low'].iloc[i-1]) / data['close'].iloc[i] >= self.min_ob_size):
                    ob_low = data['low'].iloc[i-1]
                    ob_high = data['high'].iloc[i-1]
                    order_blocks.append(('bullish', i-1, ob_low, ob_high))
            elif trend == 'bearish':
                if (data['close'].iloc[i-1] > data['open'].iloc[i-1] and
                    data['close'].iloc[i] < data['open'].iloc[i] and
                    data['close'].iloc[i] < data['open'].iloc[i-1] and
                    data['open'].iloc[i] > data['close'].iloc[i-1] and
                    (data['high'].iloc[i-1] - data['low'].iloc[i-1]) / data['close'].iloc[i] >= self.min_ob_size):
                    ob_low = data['low'].iloc[i-1]
                    ob_high = data['high'].iloc[i-1]
                    order_blocks.append(('bearish', i-1, ob_low, ob_high))
        return order_blocks

    def identify_fvgs(self, data):
        fvgs = []
        for i in range(3, len(data)):
            prev_high = data['high'].iloc[i-2]
            curr_low = data['low'].iloc[i]
            gap_size = (prev_high - curr_low) / data['close'].iloc[i]
            second_candle_body = abs(data['close'].iloc[i-1] - data['open'].iloc[i-1])
            avg_body = (abs(data['close'].iloc[i-2] - data['open'].iloc[i-2]) +
                        abs(data['close'].iloc[i] - data['open'].iloc[i])) / 2
            if (gap_size >= self.fvg_threshold and
                second_candle_body >= 2 * avg_body):
                fvgs.append(('bullish', i, curr_low, prev_high))
            prev_low = data['low'].iloc[i-2]
            curr_high = data['high'].iloc[i]
            gap_size = (curr_high - prev_low) / data['close'].iloc[i]
            if (gap_size >= self.fvg_threshold and
                second_candle_body >= 2 * avg_body):
                fvgs.append(('bearish', i, prev_low, curr_high))
        return fvgs

    def get_higher_tf_context(self, symbol, bars=100):
        mt5_tf = self.mt5_timeframes.get(self.higher_timeframe, mt5.TIMEFRAME_D1)
        data = get_data(symbol, mt5_tf, bars)
        if data is None or len(data) < 50:
            print(f"SMC: No D1 data for {symbol}")
            return None, []
        trend, swings = self.identify_market_structure(data)
        key_levels = [(s[1], s[2]) for s in swings]
        return trend, key_levels

    def get_signal(self, data, symbol=None, timeframe='M5'):
        try:
            if self.in_cooldown(symbol):
                return ('hold', 0.0)

            df = pd.DataFrame(data)
            if not all(col in df for col in ['open', 'close', 'high', 'low']):
                print(f"SMC: Missing required columns in data for {symbol} on {timeframe}")
                return ('hold', 0.0)

            # Step 1: Identify market structure
            trend, swings = self.identify_market_structure(df)
            if trend == 'neutral':
                print(f"SMC: Neutral trend for {symbol} on {timeframe}")
                return ('hold', 0.0)

            # Step 2: Get D1 context
            ht_trend, ht_levels = self.get_higher_tf_context(symbol, bars=100)
            if ht_trend is None:
                print(f"SMC: No D1 context for {symbol} on {timeframe}")
                return ('hold', 0.0)
            if (trend == 'bullish' and ht_trend != 'bullish') or (trend == 'bearish' and ht_trend != 'bearish'):
                print(f"SMC: Trend misalignment for {symbol} on {timeframe}: {timeframe}={trend}, D1={ht_trend}")
                return ('hold', 0.0)

            # Step 3: Identify liquidity zones
            equal_highs, equal_lows = self.identify_liquidity_zones(df)
            current_price = df['close'].iloc[-1]
            liquidity_near = False
            for _, high in equal_highs[-3:]:
                if abs(high - current_price) / current_price < self.liquidity_tolerance:
                    liquidity_near = True
                    break
            for _, low in equal_lows[-3:]:
                if abs(low - current_price) / current_price < self.liquidity_tolerance:
                    liquidity_near = True
                    break
            print(f"SMC: Liquidity near for {symbol} on {timeframe}: {liquidity_near}")

            # Step 4: Identify order blocks
            order_blocks = self.identify_order_blocks(df, trend, swings)
            valid_ob = None
            for ob_type, idx, ob_low, ob_high in order_blocks[-2:]:
                ob_mid = (ob_low + ob_high) / 2
                for _, level in ht_levels:
                    if abs(level - ob_mid) / level < self.liquidity_tolerance:
                        if ((trend == 'bullish' and ob_type == 'bullish' and
                             ob_low <= current_price <= ob_high) or
                            (trend == 'bearish' and ob_type == 'bearish' and
                             ob_low <= current_price <= ob_high)):
                            valid_ob = (ob_type, ob_low, ob_high)
                            break
                if valid_ob:
                    break
            if not valid_ob:
                print(f"SMC: No valid OB for {symbol} on {timeframe}")
                return ('hold', 0.0)
            print(f"SMC: Valid OB found for {symbol} on {timeframe}: {valid_ob}")

            # Step 5: Confirm with FVG
            fvgs = self.identify_fvgs(df)
            fvg_confirmed = False
            for fvg_type, idx, fvg_low, fvg_high in fvgs[-2:]:
                if ((trend == 'bullish' and fvg_type == 'bullish' and
                     fvg_low <= current_price <= fvg_high and
                     abs(fvg_low - valid_ob[1]) / current_price < 0.01) or
                    (trend == 'bearish' and fvg_type == 'bearish' and
                     fvg_low <= current_price <= fvg_high and
                     abs(fvg_high - valid_ob[2]) / current_price < 0.01)):
                    fvg_confirmed = True
                    break
            print(f"SMC: FVG confirmed for {symbol} on {timeframe}: {fvg_confirmed}")

            # Step 6: Generate signal
            strength = 0.9 if (liquidity_near and fvg_confirmed) else 0.8
            if valid_ob and (fvg_confirmed or liquidity_near):
                self.last_trade_times[symbol] = datetime.now()
                signal = 'buy' if trend == 'bullish' else 'sell'
                print(f"SMC: {signal} signal for {symbol} on {timeframe}, Strength: {strength}")
                return (signal, strength)
            print(f"SMC: No signal for {symbol} on {timeframe}: Missing FVG or liquidity")
            return ('hold', 0.0)
        except Exception as e:
            print(f"SMC: Error in get_signal for {symbol} on {timeframe}: {e}")
            return ('hold', 0.0)
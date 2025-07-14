from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from utils.mt5_connection import get_data
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from utils.data_cache import data_cache_instance
import logging

logger = logging.getLogger(__name__)

class SMCStrategy(BaseStrategy):
    """
    This strategy implements an advanced Smart Money Concept (SMC) approach.
    It moves beyond simple trend following and focuses on identifying high-probability reversals
    based on a specific sequence of market events:
    1. Liquidity Sweep: Price takes out a previous high or low.
    2. Change of Character (CHoCH): A shift in market structure immediately following the sweep.
    3. Point of Interest (POI) Mitigation: Price returns to the 'causative' Order Block or FVG
       that was responsible for the CHoCH.
    The strategy is designed to be patient and only trigger when this full narrative plays out,
    aligning with the principles from the "Advanced Smart Money Concept" guide.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name)
        # Default values are set here but will be overridden by set_config
        self.higher_timeframe = 'H4'
        self.swing_lookback = 50
        self.fvg_threshold = 0.0003
        self.liquidity_tolerance = 0.0005 # How close two highs/lows must be
        self.poi_search_range = 10 # Bars to look back for a POI after a CHoCH
        self.trade_cooldown = 30
        self.last_trade_times = {} # Stores last trade time per symbol to prevent over-trading
        self.mt5_timeframes = {
            'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1
        }
        # Initialize any other parameters from kwargs passed during creation
        self.set_config(kwargs)
        logger.debug(f"Advanced SMCStrategy initialized.")

    def set_config(self, config: dict):
        """Sets strategy parameters from a configuration dictionary."""
        self.higher_timeframe = config.get('smc_higher_timeframe', self.higher_timeframe)
        self.swing_lookback = config.get('smc_swing_lookback', self.swing_lookback)
        self.fvg_threshold = config.get('smc_fvg_threshold', self.fvg_threshold)
        self.liquidity_tolerance = config.get('smc_liquidity_tolerance', self.liquidity_tolerance)
        self.poi_search_range = config.get('smc_poi_search_range', self.poi_search_range)
        self.trade_cooldown = config.get('smc_trade_cooldown', self.trade_cooldown)
        logger.debug(f"Advanced SMCStrategy config updated: HTF={self.higher_timeframe}, Lookback={self.swing_lookback}, etc.")


    def in_cooldown(self, symbol):
        """Checks if the symbol is in a cooldown period after a trade."""
        last_trade = self.last_trade_times.get(symbol)
        if last_trade and (datetime.now(timezone.utc) - last_trade) < timedelta(minutes=self.trade_cooldown):
            logger.debug(f"SMC ({symbol}): Cooldown active.")
            return True
        return False

    def _find_swing_points(self, data: pd.DataFrame, lookback: int):
        """
        Identifies swing high and low points using a simple local max/min logic.
        This is a helper for the main market structure analysis.
        """
        swings = []
        # Use a rolling window to find local min/max which is more robust
        data['swing_high'] = data['high'].rolling(window=lookback, center=True).max()
        data['swing_low'] = data['low'].rolling(window=lookback, center=True).min()

        for i in range(len(data)):
            # A swing high is where the current high is the max in its window
            if data['high'].iloc[i] == data['swing_high'].iloc[i]:
                swings.append({'type': 'high', 'price': data['high'].iloc[i], 'index': i, 'time': data.index[i]})
            # A swing low is where the current low is the min in its window
            elif data['low'].iloc[i] == data['swing_low'].iloc[i]:
                swings.append({'type': 'low', 'price': data['low'].iloc[i], 'index': i, 'time': data.index[i]})

        # Deduplicate consecutive identical swings
        if not swings: return []
        unique_swings = [swings[0]]
        for i in range(1, len(swings)):
            if swings[i]['type'] != swings[i-1]['type']:
                unique_swings.append(swings[i])
        return unique_swings


    def analyze_market_structure(self, data: pd.DataFrame):
        """

        Analyzes the data to identify the current market structure, including trend,
        Breaks of Structure (BOS), and Changes of Character (CHoCH). This is the core
        of the new advanced logic.

        Returns:
            dict: A dictionary containing the market structure analysis, e.g.,
                  {'trend': 'bullish', 'last_event': 'BOS', 'event_index': 1480,
                   'major_high': 1.2345, 'major_low': 1.2200}
        """
        swing_points = self._find_swing_points(data, self.swing_lookback)
        if len(swing_points) < 4:
            return {'trend': 'choppy', 'last_event': None}

        # Identify HH, HL, LH, LL
        structural_points = []
        # Start with the first two points to establish a direction
        if swing_points[1]['type'] == 'high' and swing_points[0]['type'] == 'low':
             structural_points.append({'type': 'L', 'price': swing_points[0]['price'], 'index': swing_points[0]['index']})
             structural_points.append({'type': 'H', 'price': swing_points[1]['price'], 'index': swing_points[1]['index']})
        else: # Skip until we have a clear Low-High or High-Low sequence
            swing_points = swing_points[1:]
            if not swing_points or len(swing_points) < 2: return {'trend': 'choppy', 'last_event': None}
            structural_points.append({'type': 'H' if swing_points[0]['type'] == 'high' else 'L', 'price': swing_points[0]['price'], 'index': swing_points[0]['index']})
            structural_points.append({'type': 'L' if swing_points[1]['type'] == 'low' else 'H', 'price': swing_points[1]['price'], 'index': swing_points[1]['index']})


        last_h = None
        last_l = None

        for i in range(2, len(swing_points)):
            current_swing = swing_points[i]
            prev_swing = swing_points[i-1]
            prev_prev_swing = swing_points[i-2]

            if current_swing['type'] == 'high' and prev_swing['type'] == 'low':
                if current_swing['price'] > prev_prev_swing['price'] and prev_swing['price'] > swing_points[i-3]['price'] if i > 2 else True:
                   last_h = {'price': current_swing['price'], 'index': current_swing['index'], 'type': 'HH'}
                   last_l = {'price': prev_swing['price'], 'index': prev_swing['index'], 'type': 'HL'}
                elif current_swing['price'] < prev_prev_swing['price'] and prev_swing['price'] < swing_points[i-3]['price'] if i > 2 else True:
                   last_h = {'price': current_swing['price'], 'index': current_swing['index'], 'type': 'LH'}
                   last_l = {'price': prev_swing['price'], 'index': prev_swing['index'], 'type': 'LL'}


        # Now, analyze the most recent price action against the established structure
        last_major_high = max([p['price'] for p in swing_points if p['type'] == 'high'][-3:]) if swing_points else 0
        last_major_low = min([p['price'] for p in swing_points if p['type'] == 'low'][-3:]) if swing_points else 0

        # Define latest swing points for CHoCH/BOS detection
        recent_highs = [p for p in swing_points if p['type'] == 'high']
        recent_lows = [p for p in swing_points if p['type'] == 'low']

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return {'trend': 'choppy', 'last_event': None}

        last_high = recent_highs[-1]
        prev_high = recent_highs[-2]
        last_low = recent_lows[-1]
        prev_low = recent_lows[-2]

        # Determine current trend based on last two highs and lows
        trend = 'choppy'
        if last_high['price'] > prev_high['price'] and last_low['price'] > prev_low['price']:
            trend = 'bullish'
        elif last_high['price'] < prev_high['price'] and last_low['price'] < prev_low['price']:
            trend = 'bearish'

        result = {
            'trend': trend,
            'last_event': None, 'event_index': None, 'event_price': None,
            'confirmed_high': last_high if trend != 'choppy' else None,
            'confirmed_low': last_low if trend != 'choppy' else None,
            'last_swing_high': recent_highs[-1],
            'last_swing_low': recent_lows[-1],
        }

        # Detect BOS or CHoCH
        # Bullish trend: Did we break the last high (BOS) or the last low (CHoCH)?
        if trend == 'bullish':
            if data['high'].iloc[-1] > last_high['price']:
                result['last_event'] = 'BOS'
                result['event_price'] = last_high['price']
                result['event_index'] = last_high['index']
            elif data['low'].iloc[-1] < last_low['price']:
                result['last_event'] = 'CHoCH_Bearish'
                result['event_price'] = last_low['price']
                result['event_index'] = last_low['index']
        # Bearish trend: Did we break the last low (BOS) or the last high (CHoCH)?
        elif trend == 'bearish':
            if data['low'].iloc[-1] < last_low['price']:
                result['last_event'] = 'BOS'
                result['event_price'] = last_low['price']
                result['event_index'] = last_low['index']
            elif data['high'].iloc[-1] > last_high['price']:
                result['last_event'] = 'CHoCH_Bullish'
                result['event_price'] = last_high['price']
                result['event_index'] = last_high['index']

        return result


    def find_liquidity_sweep(self, data: pd.DataFrame, structure: dict):
        """Identifies if a liquidity sweep occurred just before a CHoCH."""
        if not structure.get('last_event') or 'CHoCH' not in structure['last_event']:
            return None

        choch_index = structure['event_index']
        # Look in a small window before the CHoCH
        search_window = data.iloc[max(0, choch_index - 20):choch_index]

        if structure['last_event'] == 'CHoCH_Bearish': # Bull trend ended, we swept a high
            # Find the high that was swept
            relevant_high = structure.get('confirmed_high')
            if relevant_high and not search_window.empty:
                # Check if price wicked above the high then closed below
                if (search_window['high'] > relevant_high['price']).any() and \
                   (search_window['close'] < relevant_high['price']).any():
                    logger.debug(f"Liquidity SWEEP confirmed above high at {relevant_high['price']:.4f} before bearish CHoCH.")
                    return {'type': 'high_sweep', 'price': relevant_high['price']}

        elif structure['last_event'] == 'CHoCH_Bullish': # Bear trend ended, we swept a low
            relevant_low = structure.get('confirmed_low')
            if relevant_low and not search_window.empty:
                 if (search_window['low'] < relevant_low['price']).any() and \
                    (search_window['close'] > relevant_low['price']).any():
                    logger.debug(f"Liquidity SWEEP confirmed below low at {relevant_low['price']:.4f} before bullish CHoCH.")
                    return {'type': 'low_sweep', 'price': relevant_low['price']}
        return None


    def find_causative_poi(self, data: pd.DataFrame, structure: dict):
        """
        After a CHoCH, finds the Point of Interest (Order Block or FVG)
        that caused the structural shift.
        """
        if not structure or 'CHoCH' not in (structure.get('last_event') or ''):
            return None

        choch_index = structure.get('event_index', len(data) - 1)
        # Search for the POI in the impulsive leg that led to the CHoCH
        search_start_index = max(0, choch_index - self.poi_search_range)
        search_window = data.iloc[search_start_index:choch_index + 1]

        # Find FVG (Fair Value Gaps)
        for i in range(len(search_window) - 2, 0, -1):
            c1 = search_window.iloc[i-1]
            c3 = search_window.iloc[i+1]
            current_index_in_main_df = search_start_index + i

            # Bullish FVG (for a buy setup after a bullish CHoCH)
            if structure['last_event'] == 'CHoCH_Bullish' and c3['low'] > c1['high']:
                gap_size = c3['low'] - c1['high']
                if c1['close'] != 0 and gap_size / c1['close'] >= self.fvg_threshold:
                    poi = {'type': 'bullish_fvg', 'low': c1['high'], 'high': c3['low'], 'index': current_index_in_main_df}
                    logger.debug(f"Found causative Bullish FVG POI at index {poi['index']}: [{poi['low']:.4f} - {poi['high']:.4f}]")
                    return poi

            # Bearish FVG (for a sell setup after a bearish CHoCH)
            elif structure['last_event'] == 'CHoCH_Bearish' and c1['low'] > c3['high']:
                gap_size = c1['low'] - c3['high']
                if c1['close'] != 0 and gap_size / c1['close'] >= self.fvg_threshold:
                    poi = {'type': 'bearish_fvg', 'low': c3['high'], 'high': c1['low'], 'index': current_index_in_main_df}
                    logger.debug(f"Found causative Bearish FVG POI at index {poi['index']}: [{poi['low']:.4f} - {poi['high']:.4f}]")
                    return poi

        # Find Order Block if no FVG is found
        for i in range(len(search_window) - 1, 0, -1):
            candle = search_window.iloc[i]
            prev_candle = search_window.iloc[i-1]
            current_index_in_main_df = search_start_index + i

            # Bullish OB (last down candle before the up-move causing bullish CHoCH)
            if structure['last_event'] == 'CHoCH_Bullish' and prev_candle['close'] < prev_candle['open'] and candle['close'] > candle['open']:
                poi = {'type': 'bullish_ob', 'low': prev_candle['low'], 'high': prev_candle['high'], 'index': current_index_in_main_df - 1}
                logger.debug(f"Found causative Bullish OB POI at index {poi['index']}: [{poi['low']:.4f} - {poi['high']:.4f}]")
                return poi

            # Bearish OB (last up candle before the down-move causing bearish CHoCH)
            elif structure['last_event'] == 'CHoCH_Bearish' and prev_candle['close'] > prev_candle['open'] and candle['close'] < candle['open']:
                poi = {'type': 'bearish_ob', 'low': prev_candle['low'], 'high': prev_candle['high'], 'index': current_index_in_main_df - 1}
                logger.debug(f"Found causative Bearish OB POI at index {poi['index']}: [{poi['low']:.4f} - {poi['high']:.4f}]")
                return poi

        logger.debug("No causative POI found for the recent CHoCH.")
        return None

    def get_higher_tf_context(self, symbol, bars=200):
        """Fetches and analyzes the higher timeframe for trend context."""
        htf_data = get_data(symbol, self.higher_timeframe, bars)
        if htf_data is None or len(htf_data) < self.swing_lookback * 2:
            logger.warning(f"SMC ({symbol}): Insufficient {self.higher_timeframe} data for context.")
            return 'choppy'

        htf_data.set_index('time', inplace=True)
        htf_structure = self.analyze_market_structure(htf_data)
        logger.debug(f"SMC ({symbol}): HTF ({self.higher_timeframe}) Context: Trend is {htf_structure.get('trend', 'choppy')}")
        return htf_structure.get('trend', 'choppy')


    def get_signal(self, data, symbol=None, timeframe='M5'):
        if self.in_cooldown(symbol):
            return ('hold', 0.0)

        df = pd.DataFrame(data)
        if 'time' not in df.columns or not all(c in df.columns for c in ['open','high','low','close']):
            logger.warning(f"SMC ({symbol}, {timeframe}): Data missing required columns.")
            return ('hold', 0.0)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)

        # 1. Analyze the structure on the current (lower) timeframe
        ltf_structure = self.analyze_market_structure(df)
        if not ltf_structure.get('last_event') or 'CHoCH' not in ltf_structure['last_event']:
            # The primary trigger is not present, so we wait.
            return ('hold', 0.0)

        logger.info(f"SMC ({symbol}, {timeframe}): Found trigger: {ltf_structure['last_event']} at index {ltf_structure['event_index']}")

        # 2. Find the causative Point of Interest (POI) for this CHoCH
        poi = self.find_causative_poi(df, ltf_structure)
        if not poi:
            logger.debug(f"SMC ({symbol}, {timeframe}): CHoCH occurred, but no clear causative POI found. Holding.")
            return ('hold', 0.0)

        # 3. Check if the current price is mitigating (testing) this POI
        current_price = df['close'].iloc[-1]
        is_mitigating = poi['low'] <= current_price <= poi['high']
        if not is_mitigating:
            # We have a valid setup, but price hasn't returned to our entry zone yet.
            # In a real bot, you would now 'arm' this POI and wait for price to return.
            # For this signal-based system, we just hold until it enters the zone.
            logger.debug(f"SMC ({symbol}, {timeframe}): POI identified at [{poi['low']:.4f}-{poi['high']:.4f}], but price {current_price:.4f} is not yet mitigating. Holding.")
            return ('hold', 0.0)

        logger.info(f"SMC ({symbol}, {timeframe}): Price {current_price:.4f} is MITIGATING POI at [{poi['low']:.4f}-{poi['high']:.4f}]. Proceeding with confirmations.")

        # 4. Gather Confirmations to calculate signal strength
        strength = 0.7  # Base strength for a confirmed CHoCH + POI mitigation
        signal = 'hold'

        # Confirmation A: Higher Timeframe Alignment
        htf_trend = self.get_higher_tf_context(symbol, bars=self.swing_lookback * 4)
        if (ltf_structure['last_event'] == 'CHoCH_Bullish' and htf_trend == 'bullish') or \
           (ltf_structure['last_event'] == 'CHoCH_Bearish' and htf_trend == 'bearish'):
            strength += 0.15
            logger.debug(f"SMC CONFIRMATION: HTF trend ({htf_trend}) aligns with CHoCH direction. Strength +0.15")

        # Confirmation B: Liquidity Sweep
        sweep = self.find_liquidity_sweep(df, ltf_structure)
        if sweep:
            strength += 0.15
            logger.debug(f"SMC CONFIRMATION: Liquidity sweep preceded CHoCH. Strength +0.15")

        # Final signal determination
        if ltf_structure['last_event'] == 'CHoCH_Bullish' and 'bullish' in poi['type']:
            signal = 'buy'
        elif ltf_structure['last_event'] == 'CHoCH_Bearish' and 'bearish' in poi['type']:
            signal = 'sell'
        else:
            return ('hold', 0.0)

        self.last_trade_times[symbol] = datetime.now(timezone.utc)
        logger.info(f"SMC ({symbol}, {timeframe}): FINAL SIGNAL: {signal.upper()} with strength {strength:.2f}")
        return (signal, min(strength, 1.0))


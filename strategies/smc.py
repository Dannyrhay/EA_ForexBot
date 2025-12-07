from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from utils.mt5_connection import get_data
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from utils.data_cache import data_cache_instance
import logging
from models import TradeSignal # Use new standardized signal

logger = logging.getLogger(__name__)

class SMCStrategy(BaseStrategy):
    """
    Stateful Smart Money Concept (SMC) Strategy.
    Persists market structure and Points of Interest (POIs) to avoid recalculation
    and "amnesia" between ticks.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.higher_timeframe = 'M15'
        self.swing_lookback = 25
        self.fvg_threshold = 0.0003
        self.liquidity_tolerance = 0.0005
        self.poi_search_range = 10
        self.trade_cooldown = 30
        self.min_sl_atr_multiplier = 1.0

        # State Persistence
        self.last_trade_times = {}
        self.active_pois = {} # {symbol_timeframe: [List of POI dicts]}
        self.last_analyzed_bar_time = {} # {symbol_timeframe: datetime}
        self.market_structure_cache = {} # {symbol_timeframe: structure_dict}

        self.mt5_timeframes = {
            'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1
        }
        self.set_config(kwargs)
        logger.debug(f"Stateful SMCStrategy initialized.")

    def set_config(self, config: dict):
        self.higher_timeframe = config.get('smc_higher_timeframe', self.higher_timeframe)
        self.swing_lookback = config.get('smc_swing_lookback', self.swing_lookback)
        self.fvg_threshold = config.get('smc_fvg_threshold', self.fvg_threshold)
        self.liquidity_tolerance = config.get('smc_liquidity_tolerance', self.liquidity_tolerance)
        self.poi_search_range = config.get('smc_poi_search_range', self.poi_search_range)
        self.trade_cooldown = config.get('smc_trade_cooldown', self.trade_cooldown)
        self.min_sl_atr_multiplier = config.get('min_sl_atr_multiplier', self.min_sl_atr_multiplier)

    def in_cooldown(self, symbol):
        last_trade = self.last_trade_times.get(symbol)
        if last_trade and (datetime.now(timezone.utc) - last_trade) < timedelta(minutes=self.trade_cooldown):
            return True
        return False

    def get_signal(self, data, symbol=None, timeframe='M5', **kwargs):
        """
        Main entry point. Now optimized to only run heavy analysis on new bars.
        """
        if self.in_cooldown(symbol):
            return ('hold', 0.0) # Keep legacy return format for now, or switch to TradeSignal if caller is ready

        df = pd.DataFrame(data)
        if 'time' not in df.columns or df.empty:
            return ('hold', 0.0)

        # Ensure time index
        if not isinstance(df.index, pd.DatetimeIndex):
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        df.sort_index(inplace=True)

        current_bar_time = df.index[-1]
        key = f"{symbol}_{timeframe}"

        # 1. Update State (Heavy Analysis) ONLY on New Bar
        if key not in self.last_analyzed_bar_time or current_bar_time > self.last_analyzed_bar_time[key]:
            self._update_market_structure(df, key)
            self.last_analyzed_bar_time[key] = current_bar_time

        # 2. Check for Mitigation (Light Check) EVERY Tick
        # We check if current price is inside any active POI
        current_price = df['close'].iloc[-1]
        active_pois = self.active_pois.get(key, [])

        best_setup = None

        # Iterate through POIs to find mitigation
        # We iterate backwards to prioritize recent POIs? Or just find the best one.
        for poi in active_pois:
            if self._is_mitigating(current_price, poi):
                logger.info(f"SMC ({key}): Price {current_price} mitigating POI {poi['type']} at {poi['low']}-{poi['high']}")

                # Verify Context (Liquidity Sweep, HTF) - cached in structure or re-checked?
                # For now, we re-check simple confirmations or use what's in POI metadata

                setup = self._create_trade_setup(poi, symbol, timeframe, df)
                if setup:
                    best_setup = setup
                    break # Take the first valid mitigation

        if best_setup:
            self.last_trade_times[symbol] = datetime.now(timezone.utc)
            # Return dict format for compatibility with main.py's current expectation
            # The ConsensusEngine will normalize this.
            return best_setup

        return ('hold', 0.0)

    def _update_market_structure(self, df, key):
        """
        Analyzes structure, finds CHoCH, and updates active POIs.
        """
        structure = self.analyze_market_structure(df)
        self.market_structure_cache[key] = structure

        if structure.get('last_event') and 'CHoCH' in structure['last_event']:
            logger.info(f"SMC ({key}): New Structure Event: {structure['last_event']}")

            # Find and Store POI
            poi = self.find_causative_poi(df, structure)
            if poi:
                # Add to active POIs
                if key not in self.active_pois: self.active_pois[key] = []

                # Avoid duplicates (simple check by index)
                if not any(p['index'] == poi['index'] for p in self.active_pois[key]):
                    self.active_pois[key].append(poi)
                    logger.info(f"SMC ({key}): Registered new POI: {poi}")

                    # Cleanup old POIs (keep last 5 for example)
                    if len(self.active_pois[key]) > 5:
                        self.active_pois[key].pop(0)

    def _is_mitigating(self, price, poi):
        tolerance = (poi['high'] - poi['low']) * 0.1
        return (poi['low'] - tolerance) <= price <= (poi['high'] + tolerance)

    def _create_trade_setup(self, poi, symbol, timeframe, df):
        # Calculate strength based on confirmations
        strength = 0.7

        # Check HTF Alignment (Optional: could be heavy, maybe cache this too)
        # htf_trend = self.get_higher_tf_context(symbol)
        # For speed, we might skip HTF here or assume it was checked during POI creation

        signal_type = 'buy' if 'bullish' in poi['type'] else 'sell'
        
        # Calculate ATR for Minimum SL Distance
        atr = self._calculate_atr(df)
        min_sl_dist = atr * self.min_sl_atr_multiplier if atr > 0 else 0.0

        if signal_type == 'buy':
            entry = poi['high']
            raw_sl = poi['low'] - (poi['high'] - poi['low']) * 0.1
            
            # Enforce Minimum SL Distance
            if (entry - raw_sl) < min_sl_dist:
                sl = entry - min_sl_dist
                logger.info(f"SMC: Widened SL for BUY to meet min ATR distance ({min_sl_dist:.5f})")
            else:
                sl = raw_sl

            tp = entry + (entry - sl) * 2.0
            order_type = mt5.ORDER_TYPE_BUY_LIMIT
        else:
            entry = poi['low']
            raw_sl = poi['high'] + (poi['high'] - poi['low']) * 0.1
            
            # Enforce Minimum SL Distance
            if (raw_sl - entry) < min_sl_dist:
                sl = entry + min_sl_dist
                logger.info(f"SMC: Widened SL for SELL to meet min ATR distance ({min_sl_dist:.5f})")
            else:
                sl = raw_sl

            tp = entry - (sl - entry) * 2.0
            order_type = mt5.ORDER_TYPE_SELL_LIMIT

        return {
            'signal': signal_type,
            'type': order_type,
            'entry_price': entry,
            'sl': sl,
            'tp': tp,
            'strength': strength,
            'comment': f"SMC {signal_type.upper()} Limit ({poi['type']})"
        }

    # --- Existing Helper Methods (Preserved) ---
    # _find_swing_points, _calculate_atr, analyze_market_structure, find_liquidity_sweep, find_causative_poi
    # (We need to include these methods here as we are replacing the whole file)

    def _find_swing_points(self, data: pd.DataFrame, lookback: int):
        swings = []
        data['swing_high'] = data['high'].rolling(window=lookback, center=True).max()
        data['swing_low'] = data['low'].rolling(window=lookback, center=True).min()

        for i in range(len(data)):
            if data['high'].iloc[i] == data['swing_high'].iloc[i]:
                swings.append({'type': 'high', 'price': data['high'].iloc[i], 'index': i, 'time': data.index[i]})
            elif data['low'].iloc[i] == data['swing_low'].iloc[i]:
                swings.append({'type': 'low', 'price': data['low'].iloc[i], 'index': i, 'time': data.index[i]})

        if not swings: return []
        unique_swings = [swings[0]]
        for i in range(1, len(swings)):
            if swings[i]['type'] != swings[i-1]['type']:
                unique_swings.append(swings[i])
        return unique_swings

    def _calculate_atr(self, data, period=14):
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1] if not atr.empty else 0.0

    def analyze_market_structure(self, data: pd.DataFrame):
        swing_points = self._find_swing_points(data, self.swing_lookback)
        if len(swing_points) < 4:
            return {'trend': 'choppy', 'last_event': None}

        recent_highs = [p for p in swing_points if p['type'] == 'high']
        recent_lows = [p for p in swing_points if p['type'] == 'low']

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return {'trend': 'choppy', 'last_event': None}

        last_high = recent_highs[-1]
        prev_high = recent_highs[-2]
        last_low = recent_lows[-1]
        prev_low = recent_lows[-2]

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
        }

        atr = self._calculate_atr(data)
        min_impulse_size = 0.5 * atr

        if trend == 'bullish':
            if data['high'].iloc[-1] > last_high['price']:
                impulse_size = data['high'].iloc[-1] - last_low['price']
                if impulse_size > min_impulse_size:
                    result['last_event'] = 'BOS'
                    result['event_price'] = last_high['price']
                    result['event_index'] = last_high['index']
            elif data['low'].iloc[-1] < last_low['price']:
                impulse_size = last_high['price'] - data['low'].iloc[-1]
                if impulse_size > min_impulse_size:
                    result['last_event'] = 'CHoCH_Bearish'
                    result['event_price'] = last_low['price']
                    result['event_index'] = last_low['index']

        elif trend == 'bearish':
            if data['low'].iloc[-1] < last_low['price']:
                impulse_size = last_high['price'] - data['low'].iloc[-1]
                if impulse_size > min_impulse_size:
                    result['last_event'] = 'BOS'
                    result['event_price'] = last_low['price']
                    result['event_index'] = last_low['index']
            elif data['high'].iloc[-1] > last_high['price']:
                impulse_size = data['high'].iloc[-1] - last_low['price']
                if impulse_size > min_impulse_size:
                    result['last_event'] = 'CHoCH_Bullish'
                    result['event_price'] = last_high['price']
                    result['event_index'] = last_high['index']

        return result

    def find_causative_poi(self, data: pd.DataFrame, structure: dict):
        if not structure or 'CHoCH' not in (structure.get('last_event') or ''):
            return None

        choch_index = structure.get('event_index', len(data) - 1)
        search_start_index = max(0, choch_index - self.poi_search_range)
        search_window = data.iloc[search_start_index:choch_index + 1]

        for i in range(len(search_window) - 2, 0, -1):
            c1 = search_window.iloc[i-1]
            c3 = search_window.iloc[i+1]
            current_index_in_main_df = search_start_index + i

            if structure['last_event'] == 'CHoCH_Bullish' and c3['low'] > c1['high']:
                gap_size = c3['low'] - c1['high']
                if c1['close'] != 0 and gap_size / c1['close'] >= self.fvg_threshold:
                    return {'type': 'bullish_fvg', 'low': c1['high'], 'high': c3['low'], 'index': current_index_in_main_df}

            elif structure['last_event'] == 'CHoCH_Bearish' and c1['low'] > c3['high']:
                gap_size = c1['low'] - c3['high']
                if c1['close'] != 0 and gap_size / c1['close'] >= self.fvg_threshold:
                    return {'type': 'bearish_fvg', 'low': c3['high'], 'high': c1['low'], 'index': current_index_in_main_df}

        for i in range(len(search_window) - 1, 0, -1):
            candle = search_window.iloc[i]
            prev_candle = search_window.iloc[i-1]
            current_index_in_main_df = search_start_index + i

            if structure['last_event'] == 'CHoCH_Bullish' and prev_candle['close'] < prev_candle['open'] and candle['close'] > candle['open']:
                return {'type': 'bullish_ob', 'low': prev_candle['low'], 'high': prev_candle['high'], 'index': current_index_in_main_df - 1}

            elif structure['last_event'] == 'CHoCH_Bearish' and prev_candle['close'] > prev_candle['open'] and candle['close'] < candle['open']:
                return {'type': 'bearish_ob', 'low': prev_candle['low'], 'high': prev_candle['high'], 'index': current_index_in_main_df - 1}

        return None


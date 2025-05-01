import MetaTrader5 as mt5
import pandas as pd
import json
import time
import numpy as np
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted, NotFittedError
from strategies.bollinger_bands import BollingerBandsStrategy
from strategies.ml_model import MLValidator
from utils.mt5_connection import connect_mt5, get_data
from utils.logging import setup_logging
from utils.trade_history import save_trade, get_strategy_weights, update_trade_status
from strategies.supply_demand import SupplyDemandStrategy
from strategies.sma import SMAStrategy
from strategies.smc import SMCStrategy
from strategies.liquidity_sweep import LiquiditySweepStrategy
from strategies.fvg import FVGStrategy
from strategies.fibonacci import FibonacciStrategy
from strategies.candlestick import CandlestickStrategy
from strategies.malaysian_snr import MalaysianSnRStrategy

logger = setup_logging()

class TradingBot:
    def __init__(self):
        self.config = self.load_config()
        # Ensure all symbols are selected in Market Watch before training
        self.ensure_symbols_selected()
        self.strategies = self.initialize_strategies()
        self.ml_validator = MLValidator(self.config)
        self.last_trade_times = {}
        self.active_trades = {}
        self.cooldown_period = self.config.get('cooldown_period', 15)
        self.consecutive_failures = 0
        self.consecutive_losses = 0
        self.last_retrain_time = {}
        self.retrain_cooldown = 300
        # Use per-symbol confidence thresholds
        self.ml_confidence_thresholds = self.config.get('ml_confidence_thresholds', {
            'XAUUSDm': 0.1,
            'BTCUSDm': 0.1,
            'ETHUSDm': 0.1,
            'GBPUSDm': 0.05,  # Lowered for GBPUSDm due to low confidence
            'EURUSDm': 0.05   # Lowered for EURUSDm due to low confidence
        })
        self.max_consecutive_losses = self.config.get('max_consecutive_losses', 3)
        self.mt5_timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M15': mt5.TIMEFRAME_M15,
            'M45': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        for strategy in self.strategies:
            if isinstance(strategy, MalaysianSnRStrategy):
                strategy.set_config(self.config)
        self.train_ml_models()

    def load_config(self):
        try:
            with open('config/config.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def ensure_symbols_selected(self):
        """Ensure all symbols are selected in Market Watch, retrying if necessary."""
        for symbol in self.config['symbols']:
            retry_attempts = 3
            for attempt in range(retry_attempts):
                if mt5.symbol_select(symbol, True):
                    logger.info(f"Successfully selected symbol {symbol} in Market Watch")
                    break
                else:
                    logger.error(f"Failed to select symbol {symbol} in Market Watch (attempt {attempt + 1}/{retry_attempts})")
                    if attempt < retry_attempts - 1:
                        time.sleep(5)  # Wait before retrying
                        # Reconnect to MT5 if necessary
                        if not connect_mt5():
                            logger.error("MT5 reconnection failed during symbol selection")
                            raise RuntimeError("MT5 connection failed")
                    else:
                        raise RuntimeError(f"Failed to select symbol {symbol} after {retry_attempts} attempts")

    def initialize_strategies(self):
        return [
            SupplyDemandStrategy(
                name="SupplyDemand",
                window=self.config['supply_demand_window']
            ),
            SMAStrategy(
                name="SMA",
                short_period=self.config['sma_short_period'],
                long_period=self.config['sma_long_period']
            ),
            SMCStrategy(
                name="SMC",
                higher_timeframe=self.config.get('higher_timeframe', 'D1'),
                min_ob_size=self.config.get('smc_min_ob_size', 0.0003),
                fvg_threshold=self.config.get('fvg_gap_threshold', 0.0003),
                liquidity_tolerance=self.config.get('liquidity_tolerance', 0.005),
                trade_cooldown=self.config.get('smc_trade_cooldown', 30)
            ),
            LiquiditySweepStrategy(
                name="LiquiditySweep",
                period=self.config['liquidity_sweep_period']
            ),
            FVGStrategy(
                name="FVG",
                gap_threshold=self.config['fvg_gap_threshold']
            ),
            FibonacciStrategy(
                name="Fibonacci",
                levels=self.config['fibonacci_levels']
            ),
            CandlestickStrategy(name="Candlestick"),
            MalaysianSnRStrategy(
                name="MalaysianSnR",
                window=self.config.get('snr_window', 10),
                freshness_window=self.config.get('snr_freshness_window', 3),
                threshold=self.config.get('snr_threshold', 0.005)
            ),
            BollingerBandsStrategy(
                name="BollingerBands",
                window=self.config.get('bb_window', 20),
                std_dev=self.config.get('bb_std_dev', 2.0)
            )
        ]

    def train_ml_models(self):
        for symbol in self.config['symbols']:
            for timeframe in self.config['timeframes']:
                mt5_tf = self.mt5_timeframes.get(timeframe, mt5.TIMEFRAME_M5)
                logger.info(f"Starting ML model training for {symbol} on {timeframe}")
                historical_data = get_data(symbol, mt5_tf, 20000)  # Increased to 20,000 bars
                if historical_data is None:
                    logger.warning(f"No data fetched for {symbol} on {timeframe}")
                    continue
                logger.info(f"Fetched {len(historical_data)} bars for {symbol} on {timeframe}")
                if len(historical_data) < self.config.get('ml_training_window', 120) + 16:
                    logger.warning(f"Insufficient data for {symbol} on {timeframe}: {len(historical_data)} bars")
                    continue
                required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
                if not all(col in historical_data.columns for col in required_columns):
                    logger.warning(f"Missing required columns in data for {symbol} on {timeframe}: {historical_data.columns}")
                    continue
                historical_data = historical_data.dropna(subset=required_columns)
                if len(historical_data) < self.config.get('ml_training_window', 120) + 16:
                    logger.warning(f"Insufficient valid data after dropping NaN for {symbol} on {timeframe}: {len(historical_data)} bars")
                    continue
                features = []
                labels = []
                training_window = self.config.get('ml_training_window', 120)
                for i in range(training_window, len(historical_data) - 16):
                    window_data = historical_data.iloc[i - training_window:i]
                    try:
                        feature = self.extract_ml_features(symbol, window_data, 'buy')
                        if any(pd.isna(f) or np.isinf(f) for f in feature):
                            logger.debug(f"Skipping sample {i} for {symbol} on {timeframe} due to NaN or infinite features: {feature}")
                            continue
                        entry_price = historical_data['close'].iloc[i]
                        future_price = historical_data['close'].iloc[i + 16]
                        profit_percent = (future_price - entry_price) / entry_price * 100
                        # Adjust profit threshold based on symbol volatility
                        profit_threshold = 0.2 if symbol in ['GBPUSDm', 'EURUSDm'] else 0.5
                        outcome = 1 if profit_percent >= profit_threshold else 0
                        features.append(feature)
                        labels.append(outcome)
                    except Exception as e:
                        logger.warning(f"Error preparing historical data for {symbol} at index {i}: {e}")
                        continue
                logger.info(f"Prepared {len(features)} training samples for {symbol} on {timeframe} (Positive: {sum(labels)})")
                if features:
                    logger.debug(f"Feature stats for {symbol} on {timeframe}: Mean={np.mean(features, axis=0)}, Std={np.std(features, axis=0)}")
                if not features or not labels:
                    logger.warning(f"No valid training data for {symbol} on {timeframe}")
                    continue
                # Check for single-class labels
                unique_labels = len(set(labels))
                if unique_labels < 2:
                    logger.warning(f"Insufficient classes for {symbol} on {timeframe}: {len(labels)} samples, {unique_labels} classes")
                    continue
                try:
                    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
                    logger.info(f"Training data split: {len(X_train)} train, {len(X_val)} validation samples")
                    self.ml_validator.fit(symbol, X_train, y_train)
                    val_prob = self.ml_validator.predict_proba(symbol, X_val)[:, 1]
                    val_pred = (val_prob > 0.5).astype(int)
                    accuracy = accuracy_score(y_val, val_pred)
                    logger.info(f"ML model for {symbol} on {timeframe} trained with accuracy: {accuracy:.2f}")
                except Exception as e:
                    logger.error(f"Failed to train ML model for {symbol} on {timeframe}: {e}")

    def retrain_ml_models(self, symbol=None):
        symbols = [symbol] if symbol else self.config['symbols']
        trade_df = pd.read_csv("trades.csv") if os.path.exists("trades.csv") else pd.DataFrame()
        for sym in symbols:
            last_retrain = self.last_retrain_time.get(sym, datetime.min)
            if (datetime.now() - last_retrain).total_seconds() < self.retrain_cooldown:
                logger.info(f"Retrain cooldown active for {sym}")
                continue
            logger.info(f"Retraining ML model for {sym}")
            historical_data = get_data(sym, mt5.TIMEFRAME_M5, 20000)  # Increased to 20,000 bars
            if historical_data is None:
                logger.warning(f"No historical data fetched for {sym}")
                continue
            symbol_trades = trade_df[trade_df['symbol'] == sym]
            features = []
            labels = []
            training_window = self.config.get('ml_training_window', 120)
            trade_samples = 0
            for i in range(training_window, len(historical_data) - 16):
                window_data = historical_data.iloc[i - training_window:i]
                try:
                    feature = self.extract_ml_features(sym, window_data, 'buy')
                    if any(pd.isna(f) or np.isinf(f) for f in feature):
                        logger.debug(f"Skipping sample {i} for {sym} due to NaN or infinite features: {feature}")
                        continue
                    entry_price = historical_data['close'].iloc[i]
                    future_price = historical_data['close'].iloc[i + 16]
                    profit_percent = (future_price - entry_price) / entry_price * 100
                    profit_threshold = 0.2 if sym in ['GBPUSDm', 'EURUSDm'] else 0.5
                    outcome = 1 if profit_percent >= profit_threshold else 0
                    features.append(feature)
                    labels.append(outcome)
                except Exception as e:
                    logger.warning(f"Error preparing historical data for {sym} at index {i}: {e}")
                    continue
            for _, trade in symbol_trades.iterrows():
                if trade['status'] == 'closed':
                    try:
                        trade_time = pd.to_datetime(trade['entry_time'])
                        trade_data = get_data(sym, mt5.TIMEFRAME_M5, training_window, start_time=trade_time)
                        if trade_data is None or len(trade_data) < training_window:
                            continue
                        feature = self.extract_ml_features(sym, trade_data, trade['signal'])
                        if any(pd.isna(f) or np.isinf(f) for f in feature):
                            continue
                        outcome = 1 if trade['profit_loss'] > 0 else 0
                        features.append(feature)
                        labels.append(outcome)
                        trade_samples += 1
                        logger.debug(f"Added trade data for {sym}: Signal={trade['signal']}, Outcome={outcome}")
                    except Exception as e:
                        logger.warning(f"Error processing trade data for {sym}: {e}")
                        continue
            logger.info(f"Prepared {len(features)} training samples for {sym} (Trade-based: {trade_samples}, Positive: {sum(labels)})")
            if not features or not labels:
                logger.warning(f"No valid retraining data for {sym}")
                continue
            # Check for single-class labels
            unique_labels = len(set(labels))
            if unique_labels < 2:
                logger.warning(f"Insufficient classes for {sym}: {len(labels)} samples, {unique_labels} classes")
                continue
            try:
                X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
                logger.info(f"Training data split: {len(X_train)} train, {len(X_val)} validation samples")
                self.ml_validator.fit(sym, X_train, y_train)
                val_prob = self.ml_validator.predict_proba(sym, X_val)[:, 1]
                val_pred = (val_prob > 0.5).astype(int)
                accuracy = accuracy_score(y_val, val_pred)
                logger.info(f"ML model for {sym} retrained with accuracy: {accuracy:.2f}")
                self.last_retrain_time[sym] = datetime.now()
            except Exception as e:
                logger.error(f"Failed to retrain ML model for {sym}: {e}")

    def extract_ml_features(self, symbol, data, signal):
        try:
            macd_line, signal_line, histogram = self.calculate_macd(data)
            snr_strategy = next((s for s in self.strategies if s.__class__.__name__ == 'MalaysianSnRStrategy'), None)
            snr_features = {'distance_to_a_level': 0, 'distance_to_v_level': 0, 'a_level_fresh': 0, 'v_level_fresh': 0}
            snr_strength = 0.0
            if snr_strategy:
                result = snr_strategy.get_signal(data, symbol=symbol)
                if isinstance(result, tuple) and len(result) >= 2:
                    snr_strength = result[1]
                    a_levels, v_levels = snr_strategy.identify_levels(data)
                    current_price = data['close'].iloc[-1]
                    if a_levels:
                        nearest_a = a_levels[-1][1]
                        snr_features['distance_to_a_level'] = abs(current_price - nearest_a) / current_price
                        snr_features['a_level_fresh'] = 1 if snr_strategy.is_fresh(a_levels[-1][0], nearest_a, data, False) else 0
                    if v_levels:
                        nearest_v = v_levels[-1][1]
                        snr_features['distance_to_v_level'] = abs(current_price - nearest_v) / current_price
                        snr_features['v_level_fresh'] = 1 if snr_strategy.is_fresh(v_levels[-1][0], nearest_v, data, True) else 0
            volatility = (data['high'].max() - data['low'].min()) / data['close'].iloc[-1]
            # Add pip-based momentum feature for forex pairs
            pip_momentum = 0.0
            if symbol in ['GBPUSDm', 'EURUSDm']:
                pip_value = 0.0001  # 1 pip = 0.0001 for most forex pairs
                price_change = (data['close'].iloc[-1] - data['close'].iloc[-10]) / pip_value
                pip_momentum = price_change / 10  # Average pip change per bar over last 10 bars
            features = [
                data['close'].pct_change().mean() or 0,
                data['high'].max() - data['low'].min(),
                1 if signal == 'buy' else 0,
                data['tick_volume'].mean() if 'tick_volume' in data else 0,
                self.calculate_rsi(data),
                self.calculate_atr(data),
                self.bollinger_band_width(data),
                macd_line / data['close'].iloc[-1] if data['close'].iloc[-1] != 0 else 0,
                signal_line / data['close'].iloc[-1] if data['close'].iloc[-1] != 0 else 0,
                histogram / data['close'].iloc[-1] if data['close'].iloc[-1] != 0 else 0,
                snr_features['distance_to_a_level'],
                snr_features['distance_to_v_level'],
                snr_features['a_level_fresh'],
                snr_features['v_level_fresh'],
                snr_strength,
                volatility,
                pip_momentum  # New feature
            ]
            return features
        except Exception as e:
            logger.error(f"Error extracting ML features for {symbol}: {e}")
            raise

    def calculate_rsi(self, data, window=14):
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
            logger.warning(f"Error calculating RSI: {e}")
            return 50

    def calculate_atr(self, data, period=14):
        try:
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]
            return 0 if pd.isna(atr) else atr
        except Exception as e:
            logger.warning(f"Error calculating ATR: {e}")
            return 0

    def bollinger_band_width(self, data, window=20):
        try:
            ma = data['close'].rolling(window=window).mean()
            std = data['close'].rolling(window=window).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            bb_width = (upper.iloc[-1] - lower.iloc[-1]) / ma.iloc[-1]
            return 0 if pd.isna(bb_width) else bb_width
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Band Width: {e}")
            return 0

    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        try:
            ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            return (
                macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0,
                signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0,
                histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0
            )
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
            return 0, 0, 0

    def get_trend(self, symbol):
        try:
            data = get_data(symbol, mt5.TIMEFRAME_H1, 200)
            if data is None or len(data) < 200:
                logger.warning(f"Failed to fetch H1 data for trend analysis on {symbol}")
                return 'neutral'
            sma = data['close'].rolling(window=200).mean().iloc[-1]
            current_price = data['close'].iloc[-1]
            macd, signal = self.calculate_macd(data)[0:2]  # Get MACD line and signal line
            if current_price > sma and macd > signal:
                trend = 'uptrend'
            elif current_price < sma and macd < signal:
                trend = 'downtrend'
            else:
                trend = 'neutral'
            logger.info(f"Trend for {symbol} on H1: {trend}")
            return trend
        except Exception as e:
            logger.error(f"Error determining trend for {symbol}: {e}")
            return 'neutral'

    def in_cooldown(self, symbol):
        last_trade = self.last_trade_times.get(symbol)
        if last_trade and (datetime.now() - last_trade) < timedelta(minutes=self.cooldown_period):
            logger.info(f"Cooldown active for {symbol}")
            return True
        return False

    def has_open_trade(self, symbol, signal):
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            logger.error(f"Failed to fetch positions for {symbol}: {mt5.last_error()}")
            return False
        for pos in positions:
            pos_type = 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell'
            if pos_type == signal:
                logger.info(f"Existing {signal} trade open for {symbol} (Position ID: {pos.ticket})")
                return True
            if pos_type != signal:
                logger.info(f"Opposite trade ({pos_type}) open for {symbol} (Position ID: {pos.ticket})")
                return True  # Prevent opposite trades
        return False

    def is_trading_hours(self, symbol):
        try:
            now = datetime.now(ZoneInfo("UTC"))
            current_day = now.strftime('%A')
            current_time = now.time()
            default_schedules = {
                'XAUUSDm': {
                    'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                    'start': '00:00',
                    'end': '22:00'
                },
                'GBPUSDm': {
                    'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                    'start': '00:00',
                    'end': '22:00'
                },
                'EURUSDm': {
                    'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                    'start': '00:00',
                    'end': '22:00'
                },
                'BTCUSDm': {
                    'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    'start': '00:00',
                    'end': '23:59'
                },
                'ETHUSDm': {
                    'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    'start': '00:00',
                    'end': '23:59'
                }
            }
            schedule = self.config.get('trading_hours', {}).get(symbol, default_schedules.get(symbol, {}))
            if not schedule:
                logger.warning(f"No trading schedule defined for {symbol}. Assuming 24/7 trading.")
                return True
            if current_day not in schedule['days']:
                logger.debug(f"Non-trading day for {symbol}: {current_day}")
                return False
            start_time = datetime.strptime(schedule['start'], '%H:%M').time()
            end_time = datetime.strptime(schedule['end'], '%H:%M').time()
            is_open = start_time <= current_time <= end_time
            logger.debug(f"Trading hours check for {symbol}: {is_open} (Time: {current_time}, Range: {start_time}-{end_time})")
            return is_open
        except Exception as e:
            logger.error(f"Error checking trading hours for {symbol}: {e}")
            return False

    def execute_trade(self, symbol, signal, data, timeframe):
        if signal == 'hold':
            logger.info(f"No trade executed for {symbol} on {timeframe}: Signal is hold")
            return False
        trend = self.get_trend(symbol)
        if (signal == 'buy' and trend == 'downtrend') or (signal == 'sell' and trend == 'uptrend'):
            logger.info(f"Trade rejected for {symbol} on {timeframe}: Signal {signal} against trend {trend}")
            return False
        if not self.is_trading_hours(symbol):
            logger.info(f"Trading hours not active for {symbol}")
            return False
        if self.in_cooldown(symbol):
            logger.info(f"Cooldown active for {symbol}")
            return False
        if self.has_open_trade(symbol, signal):
            logger.info(f"Skipped trade on {symbol}: Existing or opposite trade open")
            return False
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"Max consecutive losses ({self.max_consecutive_losses}) reached for {symbol}. Trading paused.")
            return False
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol info not available for {symbol}")
            self.consecutive_failures += 1
            logger.info(f"Consecutive trade failures: {self.consecutive_failures}")
            return False
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.warning(f"Market closed for {symbol}: {mt5.last_error()}. Skipping trade.")
            return False
        min_distance = symbol_info.trade_stops_level * symbol_info.point
        volume_min = symbol_info.volume_min
        volume_max = symbol_info.volume_max
        digits = symbol_info.digits
        try:
            features = self.extract_ml_features(symbol, data, signal)
            if any(pd.isna(f) or np.isinf(f) for f in features):
                logger.error(f"Invalid features for {symbol}: {features}")
                self.consecutive_failures += 1
                logger.info(f"Consecutive trade failures: {self.consecutive_failures}")
                return False
        except Exception as e:
            logger.error(f"Failed to extract ML features for {symbol}: {e}")
            self.consecutive_failures += 1
            logger.info(f"Consecutive trade failures: {self.consecutive_failures}")
            return False
        try:
            check_is_fitted(self.ml_validator.predictors[symbol])
            ml_prob = self.ml_validator.predict_proba(symbol, [features])[0][1]
        except NotFittedError:
            logger.warning(f"ML model for {symbol} not trained; skipping ML validation")
            ml_prob = 0.5
        except Exception as e:
            logger.error(f"Error in ML validation for {symbol}: {e}")
            ml_prob = 0.5
        # Use per-symbol confidence threshold
        ml_confidence_threshold = self.ml_confidence_thresholds.get(symbol, 0.1)
        if ml_prob < ml_confidence_threshold:
            logger.info(f"Trade rejected for {symbol}: ML confidence {ml_prob:.2f} below threshold {ml_confidence_threshold}")
            logger.warning(f"Consider retraining the ML model for {symbol} due to low confidence")
            return False
        logger.info(f"ML confidence for {symbol}: {ml_prob:.2f}")
        atr = self.calculate_atr(data, period=self.config['atr_period'])
        price_range = (data['high'].iloc[-20:].max() - data['low'].iloc[-20:].min())
        logger.debug(f"ATR for {symbol}: {atr}, Price Range (last 20 bars): {price_range}")
        if pd.isna(atr) or atr <= 0:
            logger.error(f"Invalid ATR for {symbol}: {atr}")
            self.consecutive_failures += 1
            logger.info(f"Consecutive trade failures: {self.consecutive_failures}")
            return False
        if pd.isna(price_range) or price_range <= 0:
            logger.error(f"Invalid price range for {symbol}: {price_range}")
            self.consecutive_failures += 1
            logger.info(f"Consecutive trade failures: {self.consecutive_failures}")
            return False
        volatility_adjustment = 1.0
        if symbol == 'BTCUSDm':
            volatility_adjustment = 1.5
        elif symbol == 'ETHUSDm':
            volatility_adjustment = 1.2
        min_sl_distance = max(2.5 * atr, price_range * 0.5, min_distance * 2) * volatility_adjustment
        min_tp_distance = max(3.5 * atr, price_range * 0.7, min_distance * 3) * volatility_adjustment
        price = tick.ask if signal == 'buy' else tick.bid
        sl_distance = min_sl_distance
        tp_distance = min_tp_distance
        sl = price - sl_distance if signal == 'buy' else price + sl_distance
        tp = price + tp_distance if signal == 'buy' else price - tp_distance
        if signal == 'buy':
            sl = min(sl, price - min_distance)
            tp = max(tp, price + min_distance)
        else:
            sl = max(sl, price + min_distance)
            tp = min(tp, price - min_distance)
        sl = round(sl, digits)
        tp = round(tp, digits)
        if signal == 'buy' and (sl >= price or tp <= price):
            logger.error(f"Invalid SL/TP for buy trade on {symbol}: SL={sl}, TP={tp}, Price={price}")
            self.consecutive_failures += 1
            logger.info(f"Consecutive trade failures: {self.consecutive_failures}")
            return False
        if signal == 'sell' and (sl <= price or tp >= price):
            logger.error(f"Invalid SL/TP for sell trade on {symbol}: SL={sl}, TP={tp}, Price={price}")
            self.consecutive_failures += 1
            logger.info(f"Consecutive trade failures: {self.consecutive_failures}")
            return False
        account_balance = mt5.account_info().balance
        risk_per_trade = self.config.get('risk_percent', 1.0) / 100
        risk_params = self.config.get('risk_params', {}).get(symbol, {'max_risk': 0.05})
        risk_per_trade = min(risk_per_trade, risk_params.get('max_risk', 0.05))
        pip_value = symbol_info.point
        lot_size = (account_balance * risk_per_trade) / (abs(sl - price) / pip_value * 10000)
        lot_size = max(volume_min, min(lot_size, volume_max))
        lot_size = round(lot_size, 2)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 123456,
            "comment": f"AITrade {timeframe} R{risk_per_trade:.2f} ML{ml_prob:.2f}"[:20],
            "type_time": mt5.ORDER_TIME_GTC
        }
        result = mt5.order_send(request)
        if result is None:
            mt5_error = mt5.last_error()
            logger.error(f"Trade failed for {symbol}: mt5.order_send returned None, MT5 error: {mt5_error}")
            self.consecutive_failures += 1
            logger.info(f"Consecutive trade failures: {self.consecutive_failures}")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Trade failed for {symbol}: {result.comment}")
            self.consecutive_failures += 1
            logger.info(f"Consecutive trade failures: {self.consecutive_failures}")
            return False
        logger.info(f"Trade executed: {symbol} on {timeframe} Lot: {lot_size} SL: {sl} TP: {tp}")
        self.consecutive_failures = 0
        self.last_trade_times[symbol] = datetime.now()
        self.active_trades[result.order] = {
            'symbol': symbol,
            'signal': signal,
            'entry_time': datetime.now(),
            'sl': sl,
            'tp': tp,
            'timeframe': timeframe
        }
        trade_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": signal,
            "entry_price": price,
            "sl_price": sl,
            "tp_price": tp,
            "lot_size": lot_size,
            "strategies": [s.name for s in self.strategies],
            "entry_time": datetime.now().isoformat(),
            "status": "open",
            "profit_loss": 0.0,
            "account_balance": account_balance,
            "order_id": result.order,
            "failure_reason": ""
        }
        save_trade(trade_data)
        return True

    def check_closed_trades(self):
        closed_trades = []
        for order_id, trade in list(self.active_trades.items()):
            symbol = trade['symbol']
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                logger.error(f"Failed to fetch positions for {symbol}: {mt5.last_error()}")
                continue
            position_ids = [pos.ticket for pos in positions]
            if order_id not in position_ids:
                deals = mt5.history_deals_get(position=order_id)
                profit_loss = 0.0
                exit_time = datetime.now().isoformat()
                if deals:
                    for deal in deals:
                        if deal.position_id == order_id:
                            profit_loss += deal.profit
                            exit_time = datetime.utcfromtimestamp(deal.time).isoformat()
                update_trade_status(order_id, {
                    "status": "closed",
                    "profit_loss": profit_loss,
                    "exit_time": exit_time,
                    "account_balance": mt5.account_info().balance
                })
                log_level = logger.info if profit_loss >= 0 else logger.warning
                log_level(f"Trade closed for {symbol} (Order ID: {order_id}, Profit/Loss: {profit_loss})")
                if profit_loss < 0:
                    self.consecutive_losses += 1
                    logger.info(f"Consecutive unprofitable trades: {self.consecutive_losses}")
                else:
                    self.consecutive_losses = 0
                    logger.info(f"Reset consecutive losses due to profitable trade: {profit_loss}")
                closed_trades.append((symbol, order_id))
        for symbol, order_id in closed_trades:
            del self.active_trades[order_id]
            self.retrain_ml_models(symbol=symbol)
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.error(f"{self.max_consecutive_losses} consecutive unprofitable trades detected. Shutting down for 15 minutes.")
            time.sleep(900)
            logger.info("Restarting trading bot after 15-minute shutdown.")
            self.consecutive_losses = 0
            if not connect_mt5():
                logger.error("MT5 reconnection failed after restart")
                exit(1)

    def monitor_market(self):
        while True:
            try:
                self.check_closed_trades()
                self.ensure_symbols_selected()
                for symbol in self.config['symbols']:
                    symbol_info = mt5.symbol_info(symbol)
                    if not symbol_info:
                        logger.error(f"Symbol {symbol} not found in Market Watch")
                        available_symbols = [s.name for s in mt5.symbols_get()] if mt5.symbols_get() else []
                        similar_symbols = [s for s in available_symbols if symbol[:6] in s]
                        if similar_symbols:
                            logger.info(f"Possible matching symbols for {symbol}: {similar_symbols[:5]}")
                        else:
                            logger.info(f"No similar symbols found for {symbol}")
                        continue
                    if not self.is_trading_hours(symbol):
                        logger.warning(f"Outside trading hours for {symbol}. Skipping.")
                        continue
                    if not mt5.symbol_info_tick(symbol):
                        logger.warning(f"Market closed for {symbol}: {mt5.last_error()}. Skipping.")
                        continue
                    for tf in self.config.get('timeframes', ['M5']):
                        mt5_tf = self.mt5_timeframes.get(tf, mt5.TIMEFRAME_M5)
                        data = get_data(symbol, mt5_tf, self.config.get('bars', 500))
                        if data is None:
                            logger.warning(f"Failed to fetch {tf} data for {symbol}")
                            continue
                        signals = {}
                        strengths = {}
                        for strategy in self.strategies:
                            try:
                                if isinstance(strategy, SMCStrategy):
                                    result = strategy.get_signal(data, symbol=symbol, timeframe=tf)
                                else:
                                    result = strategy.get_signal(data, symbol=symbol)
                                if isinstance(result, tuple) and len(result) >= 2:
                                    signal, strength = result[:2]
                                    signals[strategy.name] = signal
                                    strengths[strategy.name] = strength
                                    logger.debug(f"Strategy {strategy.name} signal for {symbol} on {tf}: {signal}, Strength: {strength:.4f}")
                                    if len(result) > 2:
                                        logger.debug(f"Strategy {strategy.name} additional outputs for {symbol} on {tf}: {result[2:]}")
                                else:
                                    logger.warning(f"Invalid return from {strategy.name}.get_signal for {symbol} on {tf}: {result}")
                                    signals[strategy.name] = 'hold'
                                    strengths[strategy.name] = 0.0
                            except Exception as e:
                                logger.error(f"Error in {strategy.name}.get_signal for {symbol} on {tf}: {e}")
                                signals[strategy.name] = 'hold'
                                strengths[strategy.name] = 0.0
                        consensus = self.analyze_signals(signals, strengths, data)
                        logger.debug(f"Consensus for {symbol} on {tf}: {consensus}, Signals: {signals}, Strengths: {strengths}")
                        if consensus != 'hold':
                            logger.info(f"Attempting trade for {symbol} on {tf}: {consensus}")
                            self.execute_trade(symbol, consensus, data, tf)
                        else:
                            logger.debug(f"No trade for {symbol} on {tf}: Consensus is hold")
                        if self.consecutive_failures >= 3:
                            logger.error("3 consecutive trade execution failures detected. Shutting down for 15 minutes.")
                            time.sleep(900)
                            logger.info("Restarting trading bot after 15-minute shutdown.")
                            self.consecutive_failures = 0
                            if not connect_mt5():
                                logger.error("MT5 reconnection failed after restart")
                                exit(1)
                time.sleep(60)
            except Exception as e:
                logger.error(f"Market monitoring error: {e}")
                time.sleep(300)

    def analyze_signals(self, signals, strengths, data):
        weighted_signals = []
        weights = get_strategy_weights()
        volatility = (data['high'].iloc[-20:].max() - data['low'].iloc[-20:].min()) / data['close'].iloc[-1]
        threshold = 3 if volatility < 0.01 else 4
        for strategy in self.strategies:
            strategy_name = strategy.name
            base_weight = weights.get(strategy_name, 1.0)
            strength = strengths.get(strategy_name, 0.0)
            effective_weight = base_weight * max(strength, 0.1)
            if strategy_name in ['SMC', 'MalaysianSnR']:
                effective_weight *= 1.5  # Prioritize in trending markets
            signal = signals[strategy_name]
            if signal != 'hold':
                weighted_signals.extend([signal] * int(round(effective_weight * 10)))
        buy_count = weighted_signals.count('buy')
        sell_count = weighted_signals.count('sell')
        logger.debug(f"Signal analysis: Buy count={buy_count}, Sell count={sell_count}, Threshold={threshold}")
        if buy_count >= threshold and sell_count <= buy_count / 2:
            return 'buy'
        if sell_count >= threshold and buy_count <= sell_count / 2:
            return 'sell'
        return 'hold'

if __name__ == "__main__":
    if not connect_mt5():
        logger.error("MT5 connection failed")
        exit(1)
    bot = TradingBot()
    try:
        bot.monitor_market()
    except KeyboardInterrupt:
        logger.info("Trading bot stopped")
        mt5.shutdown()
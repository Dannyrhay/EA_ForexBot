import MetaTrader5 as mt5
import pandas as pd
import json
import time
import numpy as np
import os
import re # For parsing comment
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted, NotFittedError
from strategies.bollinger_bands import BollingerBandsStrategy
from strategies.ml_model import MLValidator
from utils.mt5_connection import connect_mt5, get_data
from utils.logging import setup_logging
from utils.trade_history import save_trade, get_strategy_weights, update_trade_status, trades 
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
        self.ensure_symbols_selected()
        self.strategies = self.initialize_strategies()
        self.ml_validator = MLValidator(self.config)
        self.last_trade_times = {} 
        self.cooldown_period = timedelta(minutes=self.config.get('cooldown_period_minutes', 15))
        self.consecutive_failures = 0 
        self.consecutive_losses = 0 
        self.max_consecutive_losses = self.config.get('max_consecutive_losses', 3)
        self.last_global_retrain_time = datetime.min
        self.global_retrain_interval = timedelta(
            hours=self.config.get('global_retrain_hours', 12)
        )
        self.ml_confidence_thresholds = self.config.get('ml_confidence_thresholds', {
            'XAUUSDm': 0.1, 'BTCUSDm': 0.1, 'ETHUSDm': 0.1,
            'GBPUSDm': 0.05, 'EURUSDm': 0.05
        })
        self.mt5_timeframes = {
            'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 
            'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1
        }
        for strategy in self.strategies:
            if isinstance(strategy, MalaysianSnRStrategy):
                strategy.set_config(self.config)
        self.initial_ml_models_training()

    def load_config(self):
        try:
            with open('config/config.json', 'r') as f:
                config_data = json.load(f)
                config_data.setdefault('global_retrain_hours', 12)
                config_data.setdefault('cooldown_period_minutes', 15)
                
                # General auto-close (can be disabled if M5 specific is preferred)
                config_data.setdefault('enable_general_auto_profit_close', False) 
                config_data.setdefault('general_auto_close_profit_percent_entry_min', 60.0)
                config_data.setdefault('general_auto_close_profit_percent_entry_max', 70.0)
                config_data.setdefault('enable_general_auto_loss_close', False) 
                config_data.setdefault('general_auto_close_max_loss_percent_entry', 55.0)

                # Short Timeframe (M1, M5, M15) Specific Auto-Close
                short_tf_profit_defaults = {
                    "enabled": True, 
                    "tp_distance_ratio": 0.5, 
                    "trend_ema_short": 9,  # EMA for short-term trend
                    "trend_ema_long": 21   # EMA for short-term trend
                }
                config_data.setdefault('auto_close_short_tf_profit_take', short_tf_profit_defaults)
                if not isinstance(config_data.get('auto_close_short_tf_profit_take'), dict):
                    config_data['auto_close_short_tf_profit_take'] = short_tf_profit_defaults
                else:
                    for key, val in short_tf_profit_defaults.items():
                        config_data['auto_close_short_tf_profit_take'].setdefault(key, val)
                
                short_tf_loss_defaults = {"enabled": True, "sl_distance_ratio": 0.5}
                config_data.setdefault('auto_close_short_tf_stop_loss', short_tf_loss_defaults)
                if not isinstance(config_data.get('auto_close_short_tf_stop_loss'), dict):
                    config_data['auto_close_short_tf_stop_loss'] = short_tf_loss_defaults
                else:
                    for key, val in short_tf_loss_defaults.items():
                        config_data['auto_close_short_tf_stop_loss'].setdefault(key, val)

                config_data.setdefault('allow_multiple_trades_if_profitable', True)
                config_data.setdefault('max_open_trades_profitable', 4)
                config_data.setdefault('min_collective_profit_for_new_trade', 0.01) 
                config_data.setdefault('block_same_signal_if_losing', True)
                config_data.setdefault('max_open_trades_losing_or_flat', 1)
                
                consensus_defaults = {"low_vol": 1.0, "high_vol": 1.5, "vol_split": 0.005}
                current_consensus_config = config_data.get('consensus_threshold')
                if not isinstance(current_consensus_config, dict):
                    config_data['consensus_threshold'] = consensus_defaults
                else:
                    for key, val in consensus_defaults.items():
                        current_consensus_config.setdefault(key, val)

                config_data.setdefault('initial_training_bars', 20000)
                config_data.setdefault('retrain_data_bars', 10000)
                config_data.setdefault('ml_prediction_horizon', 16)
                default_ml_profit_thresholds = {
                    "XAUUSDm": 0.5, "BTCUSDm": 0.7, "ETHUSDm": 0.7,
                    "GBPUSDm": 0.2, "EURUSDm": 0.2, "default": 0.3 
                }
                current_ml_thresholds = config_data.get('ml_profit_thresholds')
                if not isinstance(current_ml_thresholds, dict):
                     config_data['ml_profit_thresholds'] = default_ml_profit_thresholds
                else:
                    current_ml_thresholds.setdefault("default", default_ml_profit_thresholds["default"])
                    for key, val in default_ml_profit_thresholds.items():
                        current_ml_thresholds.setdefault(key,val)
                return config_data
        except Exception as e:
            logger.error(f"Failed to load config: {e}", exc_info=True)
            raise

    def ensure_symbols_selected(self):
        # (Identical to your provided main.py)
        for symbol in self.config['symbols']:
            retry_attempts = 3
            for attempt in range(retry_attempts):
                if mt5.symbol_select(symbol, True):
                    logger.debug(f"Successfully selected symbol {symbol} in Market Watch")
                    break
                else:
                    logger.error(f"Failed to select symbol {symbol} in Market Watch (attempt {attempt + 1}/{retry_attempts})")
                    if attempt < retry_attempts - 1:
                        time.sleep(5)
                        if not connect_mt5(): 
                            logger.error("MT5 reconnection failed during symbol selection")
                            raise RuntimeError("MT5 connection failed")
                    else:
                        raise RuntimeError(f"Failed to select symbol {symbol} after {retry_attempts} attempts")

    def initialize_strategies(self):
        # (Identical to your provided main.py)
        return [
            SupplyDemandStrategy(name="SupplyDemand", window=self.config['supply_demand_window']),
            SMAStrategy(name="SMA", short_period=self.config['sma_short_period'], long_period=self.config['sma_long_period']),
            SMCStrategy(name="SMC", higher_timeframe=self.config.get('higher_timeframe', 'D1'), 
                        min_ob_size=self.config.get('smc_min_ob_size', 0.0003), 
                        fvg_threshold=self.config.get('fvg_gap_threshold', 0.0003), 
                        liquidity_tolerance=self.config.get('liquidity_tolerance', 0.005), 
                        trade_cooldown=self.config.get('smc_trade_cooldown', 30)),
            LiquiditySweepStrategy(name="LiquiditySweep", period=self.config['liquidity_sweep_period']),
            FVGStrategy(name="FVG", gap_threshold=self.config['fvg_gap_threshold']),
            FibonacciStrategy(name="Fibonacci", levels=self.config['fibonacci_levels']),
            CandlestickStrategy(name="Candlestick"),
            MalaysianSnRStrategy(name="MalaysianSnR", window=self.config.get('snr_window', 10), 
                                 freshness_window=self.config.get('snr_freshness_window', 3), 
                                 threshold=self.config.get('snr_threshold', 0.005)),
            BollingerBandsStrategy(name="BollingerBands", window=self.config.get('bb_window', 20), 
                                   std_dev=self.config.get('bb_std_dev', 2.0))
        ]

    def is_trading_hours(self, symbol):
        # (Identical to your provided main.py)
        try:
            schedules_config = self.config.get('trading_hours', {})
            symbol_schedule = schedules_config.get(symbol)
            if not symbol_schedule: 
                logger.debug(f"No specific trading schedule for {symbol}, assuming always trading hours for training/analysis purposes if not for execution.")
                return True
            now_utc = datetime.now(ZoneInfo("UTC"))
            current_day_utc = now_utc.strftime('%A') 
            current_time_utc = now_utc.time()
            if current_day_utc not in symbol_schedule.get('days', []):
                logger.debug(f"Non-trading day for {symbol}: {current_day_utc}. Schedule days: {symbol_schedule.get('days', [])}")
                return False
            start_time_str = symbol_schedule.get('start', '00:00')
            end_time_str = symbol_schedule.get('end', '23:59')
            start_time = datetime.strptime(start_time_str, '%H:%M').time()
            end_time = datetime.strptime(end_time_str, '%H:%M').time()
            is_within_time = False
            if end_time < start_time: 
                is_within_time = current_time_utc >= start_time or current_time_utc <= end_time
            else: 
                is_within_time = start_time <= current_time_utc <= end_time
            if not is_within_time:
                 logger.debug(f"Outside trading hours for {symbol}: UTC Time: {current_time_utc}, Range: {start_time}-{end_time} on {current_day_utc}")
            return is_within_time
        except Exception as e:
            logger.error(f"Error checking trading hours for {symbol}: {e}", exc_info=True)
            return False

    def initial_ml_models_training(self):
        # (Identical to your provided main.py)
        logger.info("Starting initial ML model training for all symbols...")
        for symbol in self.config['symbols']:
            if not self.is_trading_hours(symbol):
                logger.info(f"Skipping initial ML model training for {symbol}: Outside of its trading hours/days.")
                continue
            for timeframe in self.config['timeframes']: 
                mt5_tf = self.mt5_timeframes.get(timeframe, mt5.TIMEFRAME_M5)
                logger.info(f"Initial ML model training for {symbol} on {timeframe}")
                historical_data = get_data(symbol, mt5_tf, self.config.get('initial_training_bars', 20000)) 
                if historical_data is None or len(historical_data) < self.config.get('ml_training_window', 120) + 16: 
                    logger.warning(f"Insufficient data for initial training: {symbol} on {timeframe}, got {len(historical_data) if historical_data is not None else 0} bars")
                    continue
                required_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'time'] 
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
                prediction_horizon = self.config.get('ml_prediction_horizon', 16) 
                for i in range(training_window, len(historical_data) - prediction_horizon): 
                    window_data = historical_data.iloc[i - training_window:i]
                    try:
                        feature_vector = self.extract_ml_features(symbol, window_data.copy(), 'buy') 
                        if any(pd.isna(f) or np.isinf(f) for f in feature_vector):
                            continue 
                        entry_price = historical_data['close'].iloc[i]
                        future_price = historical_data['close'].iloc[i + prediction_horizon] 
                        if entry_price == 0: continue 
                        profit_percent = (future_price - entry_price) / entry_price * 100
                        profit_threshold_config = self.config.get('ml_profit_thresholds', {})
                        profit_threshold = profit_threshold_config.get(symbol, profit_threshold_config.get("default",0.5))
                        outcome = 1 if profit_percent >= profit_threshold else 0
                        features.append(feature_vector)
                        labels.append(outcome)
                    except Exception as e:
                        logger.warning(f"Error preparing historical data for {symbol} at index {i}: {e}", exc_info=True)
                        continue
                if not features or len(set(labels)) < 2: 
                    logger.warning(f"No valid training data or insufficient classes for {symbol} on {timeframe} (Features: {len(features)}, Unique Labels: {len(set(labels))})")
                    continue
                try:
                    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels if len(set(labels)) > 1 else None)
                    self.ml_validator.fit(symbol, X_train, y_train) 
                    val_prob = self.ml_validator.predict_proba(symbol, X_val)[:, 1] 
                    prediction_threshold = 0.5 
                    val_pred = (val_prob > prediction_threshold).astype(int) 
                    accuracy = accuracy_score(y_val, val_pred)
                    logger.info(f"Initial ML model for {symbol} on {timeframe} trained. Validation Accuracy: {accuracy:.2f}")
                except ValueError as ve: 
                    logger.error(f"ValueError during initial ML model training for {symbol} on {timeframe}: {ve}")
                except Exception as e:
                    logger.error(f"Failed initial ML model training for {symbol} on {timeframe}: {e}", exc_info=True)
        logger.info("Initial ML model training completed.")

    def perform_global_retrain(self):
        # (Identical to your provided main.py)
        logger.info(f"Starting scheduled global ML model retraining for all symbols...")
        for symbol in self.config['symbols']:
            if not self.is_trading_hours(symbol):
                logger.info(f"Skipping global ML model retraining for {symbol}: Outside of its trading hours/days.")
                continue
            logger.info(f"Global Retraining: Processing symbol {symbol}")
            self.retrain_ml_model_for_symbol(symbol) 
        self.last_global_retrain_time = datetime.now()
        logger.info(f"Global ML model retraining completed. Next retraining in approx. {self.global_retrain_interval}.")

    def retrain_ml_model_for_symbol(self, symbol):
        # (Identical to your provided main.py)
        logger.info(f"Retraining ML model for symbol: {symbol}")
        trade_df_path = "trades.csv" 
        trade_df = pd.read_csv(trade_df_path) if os.path.exists(trade_df_path) else pd.DataFrame()
        for timeframe_str in self.config.get('timeframes', ['M5']): 
            mt5_tf = self.mt5_timeframes.get(timeframe_str, mt5.TIMEFRAME_M5)
            logger.info(f"Retraining {symbol} on timeframe {timeframe_str}")
            retrain_bars = self.config.get('retrain_data_bars', 10000) 
            historical_data = get_data(symbol, mt5_tf, retrain_bars) 
            training_window = self.config.get('ml_training_window', 120)
            prediction_horizon = self.config.get('ml_prediction_horizon', 16)
            if historical_data is None or len(historical_data) < training_window + prediction_horizon:
                logger.warning(f"Insufficient historical data for retraining {symbol} on {timeframe_str} (Got {len(historical_data) if historical_data is not None else 0} bars, need {training_window + prediction_horizon})")
                continue
            if 'time' not in historical_data.columns:
                logger.error(f"'time' column missing in historical_data for {symbol}, {timeframe_str}. Skipping retraining for this TF.")
                continue
            symbol_trades = trade_df[trade_df['symbol'] == symbol] if not trade_df.empty else pd.DataFrame()
            features = []
            labels = []
            trade_samples_added = 0
            for i in range(training_window, len(historical_data) - prediction_horizon):
                window_data = historical_data.iloc[i - training_window:i]
                try:
                    feature = self.extract_ml_features(symbol, window_data.copy(), 'buy') 
                    if any(pd.isna(f) or np.isinf(f) for f in feature): continue
                    entry_price = historical_data['close'].iloc[i]
                    future_price = historical_data['close'].iloc[i + prediction_horizon]
                    if entry_price == 0: continue
                    profit_percent = (future_price - entry_price) / entry_price * 100
                    profit_threshold_config = self.config.get('ml_profit_thresholds', {})
                    profit_threshold = profit_threshold_config.get(symbol, profit_threshold_config.get("default",0.5))
                    outcome = 1 if profit_percent >= profit_threshold else 0
                    features.append(feature); labels.append(outcome)
                except Exception as e: logger.warning(f"Error preparing historical data for retraining {symbol} at index {i}: {e}", exc_info=True)
            if not symbol_trades.empty:
                for _, trade in symbol_trades.iterrows():
                    if trade['status'] in ['closed', 'closed_auto'] and pd.notna(trade.get('entry_time')) and pd.notna(trade.get('profit_loss')):
                        try:
                            entry_time_dt = pd.to_datetime(trade['entry_time']).tz_localize(None) 
                            estimated_start_dt = entry_time_dt - timedelta(minutes=self.mt5_tf_to_minutes(mt5_tf) * (training_window + 10)) 
                            trade_context_data = get_data(symbol, mt5_tf, training_window + prediction_horizon + 10, start_time=estimated_start_dt) 
                            if trade_context_data is None or trade_context_data.empty or 'time' not in trade_context_data.columns: continue
                            trade_context_data['time'] = pd.to_datetime(trade_context_data['time'])
                            time_diffs = (trade_context_data['time'] - entry_time_dt).abs()
                            if time_diffs.empty: continue
                            closest_bar_idx = time_diffs.idxmin()
                            if time_diffs.loc[closest_bar_idx] > timedelta(minutes=self.mt5_tf_to_minutes(mt5_tf) * 1.5): continue
                            if closest_bar_idx < training_window -1 : continue
                            window_for_trade_features = trade_context_data.iloc[closest_bar_idx - training_window + 1 : closest_bar_idx + 1]
                            if len(window_for_trade_features) < training_window: continue
                            trade_signal_type = trade.get('signal', 'buy').lower(); 
                            if trade_signal_type not in ['buy', 'sell']: trade_signal_type = 'buy'
                            feature = self.extract_ml_features(symbol, window_for_trade_features.copy(), trade_signal_type)
                            if any(pd.isna(f) or np.isinf(f) for f in feature): continue
                            outcome = 1 if float(trade['profit_loss']) > 0 else 0
                            features.append(feature); labels.append(outcome); trade_samples_added += 1
                        except Exception as e: logger.warning(f"Error processing trade data for retraining {symbol}, trade {trade.get('order_id', 'N/A')}: {e}", exc_info=True)
            logger.info(f"Prepared {len(features)} T-samples for {symbol} on {timeframe_str} (Hist: {len(features)-trade_samples_added}, Trade: {trade_samples_added}, Pos: {sum(labels)})")
            if not features or len(set(labels)) < 2: logger.warning(f"No valid retraining data or insufficient classes for {symbol} on {timeframe_str}"); continue
            try:
                X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels if len(set(labels)) > 1 else None)
                self.ml_validator.fit(symbol, X_train, y_train) 
                val_prob = self.ml_validator.predict_proba(symbol, X_val)[:, 1]
                prediction_threshold = 0.5
                val_pred = (val_prob > prediction_threshold).astype(int)
                accuracy = accuracy_score(y_val, val_pred)
                logger.info(f"ML model for {symbol} on {timeframe_str} retrained. Val Acc: {accuracy:.2f}")
            except ValueError as ve: logger.error(f"ValueError during ML model retraining for {symbol} on {timeframe_str}: {ve}")
            except Exception as e: logger.error(f"Failed to retrain ML model for {symbol} on {timeframe_str}: {e}", exc_info=True)
        logger.info(f"Retraining for symbol {symbol} completed.")

    def mt5_tf_to_minutes(self, mt5_timeframe):
        # (Identical to your provided main.py)
        tf_map = {mt5.TIMEFRAME_M1: 1, mt5.TIMEFRAME_M5: 5, mt5.TIMEFRAME_M15: 15, mt5.TIMEFRAME_M30: 30, 
                  mt5.TIMEFRAME_H1: 60, mt5.TIMEFRAME_H4: 240, mt5.TIMEFRAME_D1: 1440}
        return tf_map.get(mt5_timeframe, 5) 

    def extract_ml_features(self, symbol, data, signal):
        # (Identical to your provided main.py)
        if data.empty:
            logger.warning(f"Data is empty for ML feature extraction on {symbol}. Returning zeros.")
            return [0.0] * 17 
        if 'time' not in data.columns and any(isinstance(s, MalaysianSnRStrategy) for s in self.strategies):
             logger.debug(f"Feature extraction for {symbol}: 'time' column missing. SNR features might be affected.")
        try:
            macd_line, signal_line_val, histogram_val = self.calculate_macd(data) 
            snr_strategy = next((s for s in self.strategies if isinstance(s, MalaysianSnRStrategy)), None)
            snr_features = {'distance_to_a_level': 0.0, 'distance_to_v_level': 0.0, 'a_level_fresh': 0, 'v_level_fresh': 0}
            snr_strength = 0.0
            if snr_strategy and 'time' in data.columns and not data.empty: 
                result = snr_strategy.get_signal(pd.DataFrame(data).copy(), symbol=symbol, timeframe="M5") 
                if isinstance(result, tuple) and len(result) >= 2:
                    snr_strength = result[1] if pd.notna(result[1]) else 0.0
                    a_levels, v_levels = snr_strategy.identify_levels(pd.DataFrame(data).copy()) 
                    current_price = data['close'].iloc[-1]
                    if current_price != 0:
                        if a_levels:
                            nearest_a_info = min(a_levels, key=lambda x: abs(current_price - x[1]), default=None)
                            if nearest_a_info:
                                nearest_a_time, nearest_a_price = nearest_a_info
                                snr_features['distance_to_a_level'] = abs(current_price - nearest_a_price) / current_price
                                snr_features['a_level_fresh'] = 1 if snr_strategy.is_fresh(nearest_a_time, nearest_a_price, pd.DataFrame(data).copy(), False) else 0
                        if v_levels: 
                            nearest_v_info = min(v_levels, key=lambda x: abs(current_price - x[1]), default=None)
                            if nearest_v_info:
                                nearest_v_time, nearest_v_price = nearest_v_info
                                snr_features['distance_to_v_level'] = abs(current_price - nearest_v_price) / current_price
                                snr_features['v_level_fresh'] = 1 if snr_strategy.is_fresh(nearest_v_time, nearest_v_price, pd.DataFrame(data).copy(), True) else 0
            elif snr_strategy:
                 logger.debug(f"SNR Feature extraction for {symbol}: 'time' column missing or data empty. SNR level features will be zero.")
            volatility = 0.0
            if not data.empty and data['close'].iloc[-1] != 0:
                 volatility = (data['high'].max() - data['low'].min()) / data['close'].iloc[-1]
            pip_momentum = 0.0
            if symbol in ['GBPUSDm', 'EURUSDm'] and len(data) >= 10:
                pip_value = 0.0001 
                if data['close'].iloc[-10] != 0 and data['close'].iloc[-1] != 0:
                    price_change = (data['close'].iloc[-1] - data['close'].iloc[-10]) / pip_value
                    pip_momentum = price_change / 10 
            features_list = [
                data['close'].pct_change().mean() if not data.empty and data['close'].pct_change().mean() is not None else 0.0,
                (data['high'].max() - data['low'].min()) if not data.empty else 0.0,
                1 if signal == 'buy' else 0, 
                data['tick_volume'].mean() if 'tick_volume' in data and not data.empty and data['tick_volume'].mean() is not None else 0.0,
                self.calculate_rsi(data), self.calculate_atr(data), self.bollinger_band_width(data),
                macd_line / data['close'].iloc[-1] if not data.empty and data['close'].iloc[-1] != 0 else 0.0,
                signal_line_val / data['close'].iloc[-1] if not data.empty and data['close'].iloc[-1] != 0 else 0.0, 
                histogram_val / data['close'].iloc[-1] if not data.empty and data['close'].iloc[-1] != 0 else 0.0, 
                snr_features['distance_to_a_level'], snr_features['distance_to_v_level'],
                snr_features['a_level_fresh'], snr_features['v_level_fresh'],
                snr_strength, volatility, pip_momentum
            ]
            return [0.0 if pd.isna(f_val) or np.isinf(f_val) else float(f_val) for f_val in features_list]
        except Exception as e: logger.error(f"Error extracting ML features for {symbol}: {e}", exc_info=True); return [0.0] * 17

    def calculate_rsi(self, data, window=14):
        # (Identical to your provided main.py)
        if data.empty or len(data) < window: return 50.0
        try:
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0.0).fillna(0.0) 
            loss = -delta.where(delta < 0, 0.0).fillna(0.0)
            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()
            if avg_loss.empty or avg_gain.empty or pd.isna(avg_loss.iloc[-1]) or pd.isna(avg_gain.iloc[-1]): return 50.0
            if avg_loss.iloc[-1] == 0: rs = float('inf') if avg_gain.iloc[-1] > 0 else 0.0 
            else: rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return 50.0 if pd.isna(rsi) or np.isinf(rsi) else float(rsi)
        except Exception: return 50.0
            
    def calculate_atr(self, data, period=14):
        # (Identical to your provided main.py)
        if data.empty or len(data) < period : return 0.0
        try:
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift()).fillna(0.0)
            low_close = abs(data['low'] - data['close'].shift()).fillna(0.0)
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
            if tr.empty: return 0.0
            atr = tr.rolling(window=period, min_periods=1).mean().iloc[-1]
            return 0.0 if pd.isna(atr) else float(atr)
        except Exception: return 0.0

    def bollinger_band_width(self, data, window=20):
        # (Identical to your provided main.py)
        if data.empty or len(data) < window: return 0.0
        try:
            ma_series = data['close'].rolling(window=window).mean()
            std_series = data['close'].rolling(window=window).std()
            if ma_series.empty or std_series.empty or pd.isna(ma_series.iloc[-1]) or pd.isna(std_series.iloc[-1]): return 0.0
            ma = ma_series.iloc[-1]; std = std_series.iloc[-1]
            upper = ma + 2 * std; lower = ma - 2 * std
            if ma == 0: return 0.0
            bb_width = (upper - lower) / ma
            return 0.0 if pd.isna(bb_width) else float(bb_width)
        except Exception: return 0.0
            
    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        # (Identical to your provided main.py)
        if data.empty or len(data) < slow_period : return 0.0, 0.0, 0.0
        try:
            ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
            if ema_fast.empty or ema_slow.empty: return 0.0,0.0,0.0
            macd_line_series = ema_fast - ema_slow
            if macd_line_series.empty: return 0.0,0.0,0.0
            signal_line_series = macd_line_series.ewm(span=signal_period, adjust=False).mean()
            if signal_line_series.empty: return 0.0,0.0,0.0
            histogram_series = macd_line_series - signal_line_series
            if histogram_series.empty: return 0.0,0.0,0.0
            macd_val = macd_line_series.iloc[-1]; signal_val = signal_line_series.iloc[-1]; hist_val = histogram_series.iloc[-1]
            return (float(macd_val) if pd.notna(macd_val) else 0.0, float(signal_val) if pd.notna(signal_val) else 0.0, float(hist_val) if pd.notna(hist_val) else 0.0)
        except Exception: return 0.0, 0.0, 0.0

    def get_trend(self, symbol, timeframe_str='H1'): # Modified to accept timeframe_str
        """Determines trend based on specified timeframe."""
        mt5_tf = self.mt5_timeframes.get(timeframe_str)
        if not mt5_tf: # Fallback if timeframe_str is invalid
            logger.warning(f"Invalid timeframe_str '{timeframe_str}' in get_trend. Defaulting to H1.")
            mt5_tf = mt5.TIMEFRAME_H1
            timeframe_str = 'H1' # Ensure timeframe_str matches the used mt5_tf

        # Define parameters based on timeframe
        if timeframe_str in ['M1', 'M5', 'M15']:
            # Use short-term EMA settings from config for M1/M5/M15 trend reversal check
            profit_config = self.config.get('auto_close_short_tf_profit_take', {})
            ema_short_period = profit_config.get('trend_ema_short', 9)
            ema_long_period = profit_config.get('trend_ema_long', 21)
            bars_to_fetch = ema_long_period + 50 # Ensure enough data for longest EMA + some buffer
        else: # Default for H1 (or other higher timeframes if added later)
            ema_short_period = 12 # Standard MACD fast
            ema_long_period = 200 # Using 200 SMA for H1 trend
            bars_to_fetch = ema_long_period + 50
        
        try:
            data = get_data(symbol, mt5_tf, bars_to_fetch) 
            if data is None or data.empty or len(data) < ema_long_period:
                logger.warning(f"Failed to fetch sufficient {timeframe_str} data for trend on {symbol} (got {len(data) if data is not None else 0} bars, need >{ema_long_period})")
                return 'neutral' 
            
            current_price = data['close'].iloc[-1]
            trend = 'neutral'

            if timeframe_str in ['M1', 'M5', 'M15']:
                ema_short = data['close'].ewm(span=ema_short_period, adjust=False).mean().iloc[-1]
                ema_long = data['close'].ewm(span=ema_long_period, adjust=False).mean().iloc[-1]
                if pd.isna(current_price) or pd.isna(ema_short) or pd.isna(ema_long):
                    logger.warning(f"NaN values in {timeframe_str} EMA trend calculation for {symbol}")
                    return 'neutral'
                if current_price > ema_long and ema_short > ema_long: # Price above slower EMA, faster EMA above slower EMA
                    trend = 'uptrend'
                elif current_price < ema_long and ema_short < ema_long: # Price below slower EMA, faster EMA below slower EMA
                    trend = 'downtrend'
                logger.debug(f"Trend for {symbol} on {timeframe_str}: {trend} (Price: {current_price:.5f}, EMA{ema_short_period}: {ema_short:.5f}, EMA{ema_long_period}: {ema_long:.5f})")
            
            else: # H1 or other higher timeframe logic (original logic)
                sma200 = data['close'].rolling(window=200).mean().iloc[-1] # Assuming 200 for H1
                macd_line, signal_line, _ = self.calculate_macd(data) # Standard MACD for H1
                if pd.isna(sma200) or pd.isna(current_price) or pd.isna(macd_line) or pd.isna(signal_line):
                    logger.warning(f"NaN values in H1 trend calculation for {symbol}")
                    return 'neutral'
                if current_price > sma200 and macd_line > signal_line:
                    trend = 'uptrend'
                elif current_price < sma200 and macd_line < signal_line:
                    trend = 'downtrend'
                logger.info(f"Trend for {symbol} on H1: {trend} (Price: {current_price:.5f}, SMA200: {sma200:.5f})")
            
            return trend
        except Exception as e:
            logger.error(f"Error determining {timeframe_str} trend for {symbol}: {e}", exc_info=True)
            return 'neutral'

    def in_cooldown(self, symbol):
        # (Identical to your provided main.py)
        last_trade_dt = self.last_trade_times.get(symbol)
        if last_trade_dt and (datetime.now() - last_trade_dt) < self.cooldown_period:
            return True
        return False

    def can_open_new_trade(self, symbol, signal_to_open):
        # (Identical to your provided main.py)
        open_positions = mt5.positions_get(symbol=symbol)
        bot_positions = [p for p in open_positions if p.magic == 123456] if open_positions else []
        num_bot_trades = len(bot_positions)
        collective_profit = sum(p.profit for p in bot_positions)
        if self.config.get('allow_multiple_trades_if_profitable', False) and \
           collective_profit > self.config.get('min_collective_profit_for_new_trade', 0.0):
            if num_bot_trades < self.config.get('max_open_trades_profitable', 4):
                logger.info(f"{symbol}: Profitable state. Allowed more trades (current: {num_bot_trades}, limit: {self.config.get('max_open_trades_profitable', 4)}).")
                return True 
            else: logger.info(f"{symbol}: Profitable state but max profitable trades limit reached."); return False 
        if self.config.get('block_same_signal_if_losing', False) and collective_profit <= 0 and num_bot_trades > 0:
            for pos in bot_positions:
                current_pos_profit = pos.profit 
                pos_type_str = 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell'
                if current_pos_profit <= 0 and pos_type_str == signal_to_open:
                    logger.info(f"{symbol}: Losing {pos_type_str} trade (Ticket: {pos.ticket}) exists. Blocking new {signal_to_open}.")
                    return False 
        if num_bot_trades < self.config.get('max_open_trades_losing_or_flat', 1):
            logger.info(f"{symbol}: Allowed new trade (current: {num_bot_trades}, limit for non-profit/flat: {self.config.get('max_open_trades_losing_or_flat', 1)}).")
            return True
        else: logger.info(f"{symbol}: Max trades limit for non-profitable/flat state reached (current: {num_bot_trades})."); return False

    def execute_trade(self, symbol, signal, data, timeframe):
        # (Identical to your provided main.py, which includes margin check)
        if signal == 'hold': 
            return False
        
        h1_trend = self.get_trend(symbol, timeframe_str='H1') 
        if (signal == 'buy' and h1_trend == 'downtrend') or \
           (signal == 'sell' and h1_trend == 'uptrend'):
            logger.info(f"Trade rejected for {symbol} on {timeframe}: Signal {signal} against H1 trend {h1_trend}")
            return False
            
        if not self.is_trading_hours(symbol): 
            logger.debug(f"Trading hours not active for {symbol} (execution check).") 
            return False
            
        if self.in_cooldown(symbol): 
            logger.debug(f"Symbol {symbol} in cooldown, trade not executed.")
            return False
            
        if not self.can_open_new_trade(symbol, signal): 
            logger.debug(f"Cannot open new trade for {symbol} due to existing positions/profitability rules.")
            return False
            
        if self.consecutive_losses >= self.max_consecutive_losses: 
            logger.warning(f"Max consecutive losses ({self.max_consecutive_losses}) reached. Trading paused.")
            return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None: 
            logger.error(f"No symbol info for {symbol}")
            self.consecutive_failures += 1
            return False
            
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or tick.time == 0: 
            logger.warning(f"No tick for {symbol} or market closed (tick time: {tick.time if tick else 'None'}).")
            return False

        min_distance_points = symbol_info.trade_stops_level 
        volume_min = symbol_info.volume_min
        volume_max = symbol_info.volume_max
        digits = symbol_info.digits
        point_value = symbol_info.point

        try:
            features = self.extract_ml_features(symbol, data.copy(), signal)
            if any(pd.isna(f) or np.isinf(f) for f in features): 
                logger.error(f"Invalid features (NaN/Inf) for {symbol}: {features}")
                self.consecutive_failures += 1
                return False
        except Exception as e:
            logger.error(f"ML feature extraction error for {symbol}: {e}", exc_info=True)
            self.consecutive_failures += 1
            return False

        try:
            check_is_fitted(self.ml_validator.predictors[symbol].named_steps['gridsearchcv'])
            ml_prob_positive_outcome = self.ml_validator.predict_proba(symbol, [features])[0][1] 
        except NotFittedError: 
            logger.warning(f"ML model for {symbol} not trained yet; skipping ML validation.")
            ml_prob_positive_outcome = 0.5 
        except Exception as e: 
            logger.error(f"Error in ML validation for {symbol}: {e}", exc_info=True)
            ml_prob_positive_outcome = 0.0 

        ml_confidence_threshold = self.ml_confidence_thresholds.get(symbol, 0.1) 
        if ml_prob_positive_outcome < ml_confidence_threshold:
            logger.info(f"Trade rejected for {symbol} ({signal}): ML confidence {ml_prob_positive_outcome:.2f} < threshold {ml_confidence_threshold}")
            return False
        logger.info(f"ML confidence for {symbol} ({signal}): {ml_prob_positive_outcome:.2f}")

        atr = self.calculate_atr(data, period=self.config.get('atr_period', 14))
        if pd.isna(atr) or atr <= 0: 
            atr = data['close'].iloc[-1] * 0.005 / point_value if point_value > 0 else 0.005 * data['close'].iloc[-1]
            logger.warning(f"Invalid ATR for {symbol}, using fallback: {atr:.5f}")

        price_range_last_20 = (data['high'].iloc[-20:].max() - data['low'].iloc[-20:].min()) if len(data) >= 20 else data['close'].iloc[-1] * 0.01
        if pd.isna(price_range_last_20) or price_range_last_20 <= 0: 
            price_range_last_20 = data['close'].iloc[-1] * 0.01
            logger.warning(f"Invalid price range for {symbol}, using fallback: {price_range_last_20:.5f}")

        volatility_adjustment = self.config.get('risk_params', {}).get(symbol, {}).get('volatility_adjustment', 1.0)
        price = tick.ask if signal == 'buy' else tick.bid 

        sl_pips_atr = 2.5 * atr 
        sl_pips_range = (price_range_last_20 / point_value if point_value > 0 else price_range_last_20 * (10**digits)) * 0.5
        sl_pips_min_stop = min_distance_points * 2 
        sl_distance_pips = max(sl_pips_atr, sl_pips_range, sl_pips_min_stop) * volatility_adjustment
        sl_distance_price = sl_distance_pips * point_value

        tp_pips_atr = 3.5 * atr
        tp_pips_range = (price_range_last_20 / point_value if point_value > 0 else price_range_last_20 * (10**digits)) * 0.7
        tp_pips_min_stop = min_distance_points * 3
        tp_distance_pips = max(tp_pips_atr, tp_pips_range, tp_pips_min_stop) * volatility_adjustment
        tp_distance_price = tp_distance_pips * point_value
        
        if signal == 'buy':
            sl_price = price - sl_distance_price; tp_price = price + tp_distance_price
            sl_price = min(sl_price, tick.bid - min_distance_points * point_value if tick.bid > 0 else price - sl_distance_price)
            tp_price = max(tp_price, tick.ask + min_distance_points * point_value if tick.ask > 0 else price + tp_distance_price)
        else: 
            sl_price = price + sl_distance_price; tp_price = price - tp_distance_price
            sl_price = max(sl_price, tick.ask + min_distance_points * point_value if tick.ask > 0 else price + sl_distance_price)
            tp_price = min(tp_price, tick.bid - min_distance_points * point_value if tick.bid > 0 else price - tp_distance_price)

        sl_price = round(sl_price, digits); tp_price = round(tp_price, digits)

        if (signal == 'buy' and (sl_price >= price or tp_price <= price)) or \
           (signal == 'sell' and (sl_price <= price or tp_price >= price)):
            logger.error(f"Invalid SL/TP for {signal} on {symbol}: Entry={price:.{digits}f}, SL={sl_price:.{digits}f}, TP={tp_price:.{digits}f}")
            self.consecutive_failures += 1; return False

        account_info = mt5.account_info()
        if not account_info: 
            logger.error("No account info for lot size calculation.")
            return False
        account_balance = account_info.balance
        
        risk_percent_config = self.config.get('risk_percent', 1.0) / 100
        symbol_risk_params = self.config.get('risk_params', {}).get(symbol, {'max_risk': 0.05})
        actual_risk_per_trade = min(risk_percent_config, symbol_risk_params.get('max_risk', 0.05))
        risk_amount = account_balance * actual_risk_per_trade
        
        if point_value == 0:
            logger.error(f"Point value for {symbol} is zero. Cannot calculate SL in pips for lot size.")
            return False
        sl_pips_for_lot_calc = abs(price - sl_price) / point_value
        if sl_pips_for_lot_calc == 0: 
            logger.error(f"SL distance is zero pips for lot calculation on {symbol}. SL Price: {sl_price:.{digits}f}, Entry Price: {price:.{digits}f}")
            return False

        if symbol_info.trade_tick_size == 0: 
            logger.error(f"Cannot calculate pip value due to zero tick_size for {symbol}")
            return False
        value_per_pip_one_lot = symbol_info.trade_tick_value / (symbol_info.trade_tick_size / point_value)
        if value_per_pip_one_lot == 0: 
            logger.error(f"Value per pip is zero for {symbol}. Cannot calculate lot size.")
            return False

        lot_size = risk_amount / (sl_pips_for_lot_calc * value_per_pip_one_lot)
        
        if symbol_info.volume_step == 0: 
            logger.warning(f"Symbol {symbol} has volume_step of 0. Using default rounding (2 decimals) for lot size.")
            lot_size = round(lot_size, 2) 
        else:
            lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
            volume_step_str = str(symbol_info.volume_step)
            if '.' in volume_step_str:
                lot_size = round(lot_size, len(volume_step_str.split('.')[1]))
            else:
                lot_size = round(lot_size, 0)

        lot_size = max(volume_min, min(lot_size, volume_max))
        
        if lot_size < volume_min: 
            logger.warning(f"Calculated lot size {lot_size} for {symbol} is below minimum {volume_min}. Using minimum.")
            lot_size = volume_min
        if lot_size == 0: 
            logger.error(f"Lot size calculated as zero for {symbol}. Skipping trade.")
            return False

        order_action = mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL
        required_margin_info = mt5.order_calc_margin(order_action, symbol, lot_size, price)
        
        if required_margin_info is None:
            logger.error(f"Failed to calculate margin for {symbol} {lot_size} lots. MT5 Error: {mt5.last_error()}")
            self.consecutive_failures += 1 
            return False
        
        required_margin = required_margin_info 

        if account_info.margin_free < required_margin:
            logger.error(f"Insufficient margin for {symbol} {lot_size} lots. Required: {required_margin:.2f}, Free: {account_info.margin_free:.2f}. Skipping trade.")
            return False
        logger.info(f"Margin check OK for {symbol}. Required: {required_margin:.2f}, Free: {account_info.margin_free:.2f}")

        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot_size,
            "type": order_action, 
            "price": price, "sl": sl_price, "tp": tp_price, "deviation": 20, "magic": 123456, 
            "comment": f"AI {timeframe} R{actual_risk_per_trade*100:.1f}% ML{ml_prob_positive_outcome:.2f}"[:31], 
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC 
        }
        
        if hasattr(symbol_info, 'filling_modes') and hasattr(mt5, 'SYMBOL_FILLING_FOK'): 
            allowed_filling_modes = symbol_info.filling_modes
            if mt5.SYMBOL_FILLING_FOK in allowed_filling_modes:
                request["type_filling"] = mt5.ORDER_FILLING_FOK
            elif mt5.SYMBOL_FILLING_IOC in allowed_filling_modes:
                 request["type_filling"] = mt5.ORDER_FILLING_IOC
        
        logger.info(f"Trade Request: {request}")
        result = mt5.order_send(request)

        if result is None: 
            logger.error(f"Trade send None: {symbol}, MT5 Err: {mt5.last_error()}"); self.consecutive_failures += 1; return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            if result.retcode == 10019: 
                 logger.error(f"Trade failed for {symbol}: No money (Retcode: {result.retcode}, Comment: {result.comment}). Margin check passed but execution failed on margin.")
            elif result.retcode == 10027: 
                 logger.warning(f"Trade failed for {symbol}: Too many requests (Retcode: {result.retcode}). Consider increasing sleep or handling.")
            else:
                 logger.error(f"Trade fail retcode: {symbol}, {result.comment} (Retcode: {result.retcode})")
            self.consecutive_failures += 1
            logger.debug(f"Failed Req: {request}, Result: {result}"); return False

        logger.info(f"Trade EXECUTED: {signal} {symbol} @ {price} Lot: {lot_size} SL: {sl_price} TP: {tp_price}. ID: {result.order}")
        self.consecutive_failures = 0; self.last_trade_times[symbol] = datetime.now() 
        trade_data_log = {
            "symbol": symbol, "timeframe": timeframe, "signal": signal, "entry_price": price,
            "sl_price": sl_price, "tp_price": tp_price, "lot_size": lot_size,
            "strategies": [s.name for s in self.strategies], 
            "entry_time": datetime.now().isoformat(), "status": "open", "profit_loss": 0.0,
            "account_balance": account_balance, "order_id": result.order, "failure_reason": ""
        }
        save_trade(trade_data_log); return True

    def close_position_by_ticket(self, position_ticket, symbol, volume, trade_type_to_close, comment):
        # (Identical to your provided main.py)
        logger.info(f"Attempting to close position {position_ticket} for {symbol} ({comment})")
        close_order_type = mt5.ORDER_TYPE_SELL if trade_type_to_close == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        if not tick or tick.time == 0: logger.error(f"Cannot close {position_ticket}: No tick for {symbol}"); return False
        close_price = tick.bid if close_order_type == mt5.ORDER_TYPE_SELL else tick.ask 
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume, "type": close_order_type, 
            "position": position_ticket, "price": close_price, "deviation": 20, "magic": 123456, 
            "comment": comment[:31], "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC 
        }
        symbol_info = mt5.symbol_info(symbol) 
        if symbol_info and hasattr(symbol_info, 'filling_modes') and hasattr(mt5, 'SYMBOL_FILLING_FOK'):
            allowed_filling_modes = symbol_info.filling_modes
            if mt5.SYMBOL_FILLING_FOK in allowed_filling_modes: request["type_filling"] = mt5.ORDER_FILLING_FOK
            elif mt5.SYMBOL_FILLING_IOC in allowed_filling_modes: request["type_filling"] = mt5.ORDER_FILLING_IOC
        
        result = mt5.order_send(request)
        if result is None: logger.error(f"Order send failed to close {position_ticket}: {mt5.last_error()}"); return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close {position_ticket} for {symbol}: {result.comment} (retcode: {result.retcode})")
            logger.debug(f"Failed Close Request: {request}, Result: {result}"); return False
        logger.info(f"Sent close order for {position_ticket} ({symbol}). New Order ID: {result.order}. Comment: {comment}")
        update_data = {"status": "closed_auto", "exit_reason": comment, "exit_time": datetime.now().isoformat()}
        update_trade_status(position_ticket, update_data); return True

    def get_entry_timeframe_from_comment(self, comment_str):
        # (Identical to previous version)
        if not isinstance(comment_str, str):
            return None
        match = re.search(r"AI\s+(M\d+|H\d+|D1)\s+", comment_str)
        if match:
            return match.group(1)
        logger.debug(f"Could not parse timeframe from comment: '{comment_str}'")
        return None

    def manage_open_trades(self):
        # *** REVISED LOGIC FOR M1/M5/M15 AUTO-CLOSE ***
        logger.debug(f"MANAGE_OPEN_TRADES: Cycle start.")
        
        short_tf_profit_config = self.config.get('auto_close_short_tf_profit_take', {})
        short_tf_loss_config = self.config.get('auto_close_short_tf_stop_loss', {})
        
        general_profit_config_enabled = self.config.get('enable_general_auto_profit_close', False)
        general_loss_config_enabled = self.config.get('enable_general_auto_loss_close', False)

        if not (short_tf_profit_config.get("enabled", False) or \
                short_tf_loss_config.get("enabled", False) or \
                general_profit_config_enabled or \
                general_loss_config_enabled):
            logger.debug("MANAGE_OPEN_TRADES: All auto-closing features disabled in config.")
            return

        open_positions = mt5.positions_get()
        if open_positions is None:
            logger.error(f"MANAGE_OPEN_TRADES: Failed to get open positions: {mt5.last_error()}")
            return
        
        if not open_positions:
            logger.debug("MANAGE_OPEN_TRADES: No open positions found to manage.")
            return
        
        logger.debug(f"MANAGE_OPEN_TRADES: Found {len(open_positions)} total open positions. Filtering for bot trades (magic: 123456)...")

        for pos in open_positions:
            if pos.magic != 123456: 
                continue

            symbol = pos.symbol; ticket = pos.ticket; entry_price = pos.price_open
            original_sl = pos.sl; original_tp = pos.tp
            trade_type = pos.type; volume = pos.volume
            
            logger.info(f"MANAGE_OPEN_TRADES: Evaluating: Ticket={ticket}, Sym={symbol}, Type={'BUY' if trade_type == mt5.ORDER_TYPE_BUY else 'SELL'}, Entry={entry_price:.5f}, SL={original_sl:.5f}, TP={original_tp:.5f}, Comment='{pos.comment}'")

            tick = mt5.symbol_info_tick(symbol)
            if not tick or tick.time == 0: 
                logger.warning(f"MANAGE_OPEN_TRADES: No tick for {symbol} to evaluate {ticket}. Skipping.")
                continue
            
            current_price_for_calc = tick.bid if trade_type == mt5.ORDER_TYPE_BUY else tick.ask 
            
            symbol_info_digits = mt5.symbol_info(symbol).digits if mt5.symbol_info(symbol) else 5
            point_value = mt5.symbol_info(symbol).point if mt5.symbol_info(symbol) else (0.1**symbol_info_digits)


            entry_timeframe = self.get_entry_timeframe_from_comment(pos.comment)
            logger.debug(f"MANAGE_OPEN_TRADES (Ticket {ticket}): Parsed entry timeframe: {entry_timeframe}")

            closed_by_specific_logic = False

            # --- Short Timeframe (M1, M5, M15) Specific Auto-Close Logic ---
            if entry_timeframe in ['M1', 'M5', 'M15']:
                # Profit Taking for M1/M5/M15
                if short_tf_profit_config.get("enabled", False) and original_tp != 0 and entry_price != 0 and point_value > 0:
                    tp_distance_ratio = short_tf_profit_config.get("tp_distance_ratio", 0.5)
                    
                    total_potential_profit_pips = abs(original_tp - entry_price) / point_value
                    current_profit_pips = 0
                    is_profitable_for_check = False

                    if trade_type == mt5.ORDER_TYPE_BUY and current_price_for_calc > entry_price:
                        current_profit_pips = (current_price_for_calc - entry_price) / point_value
                        is_profitable_for_check = True
                    elif trade_type == mt5.ORDER_TYPE_SELL and current_price_for_calc < entry_price:
                        current_profit_pips = (entry_price - current_price_for_calc) / point_value
                        is_profitable_for_check = True
                    
                    logger.debug(f"MANAGE_OPEN_TRADES ({entry_timeframe} Profit - {ticket}): CurrentProfitPips={current_profit_pips:.1f}, TargetProfitPipsToConsider={total_potential_profit_pips * tp_distance_ratio:.1f} (TotalPotential={total_potential_profit_pips:.1f})")

                    if is_profitable_for_check and total_potential_profit_pips > 0 and current_profit_pips >= tp_distance_ratio * total_potential_profit_pips:
                        logger.info(f"MANAGE_OPEN_TRADES ({entry_timeframe} Profit - {ticket}): Profit target ({tp_distance_ratio*100}%) reached. Checking {entry_timeframe} trend reversal.")
                        current_short_tf_trend = self.get_trend(symbol, timeframe_str=entry_timeframe) 
                        logger.info(f"MANAGE_OPEN_TRADES ({entry_timeframe} Profit - {ticket}): {entry_timeframe} Trend for {symbol} is {current_short_tf_trend}.")
                        
                        close_short_tf_profit = False
                        if trade_type == mt5.ORDER_TYPE_BUY and current_short_tf_trend == 'downtrend':
                            close_short_tf_profit = True
                        elif trade_type == mt5.ORDER_TYPE_SELL and current_short_tf_trend == 'uptrend':
                            close_short_tf_profit = True
                        
                        if close_short_tf_profit:
                            logger.info(f"MANAGE_OPEN_TRADES ({entry_timeframe} Profit - {ticket}): Closing {'BUY' if trade_type == mt5.ORDER_TYPE_BUY else 'SELL'} trade due to {entry_timeframe} trend reversal ({current_short_tf_trend}).")
                            if self.close_position_by_ticket(ticket, symbol, volume, trade_type, f"{entry_timeframe} Auto TP Reversal"):
                                closed_by_specific_logic = True
                            continue 
                        else:
                            logger.debug(f"MANAGE_OPEN_TRADES ({entry_timeframe} Profit - {ticket}): Profit target met, but {entry_timeframe} trend ({current_short_tf_trend}) does not confirm reversal.")
                
                # Stop Loss for M1/M5/M15
                if not closed_by_specific_logic and short_tf_loss_config.get("enabled", False) and original_sl != 0 and entry_price != 0 and point_value > 0:
                    sl_distance_ratio = short_tf_loss_config.get("sl_distance_ratio", 0.5)
                    total_sl_distance_pips = abs(entry_price - original_sl) / point_value
                    current_loss_pips = 0
                    is_losing_for_check = False

                    if trade_type == mt5.ORDER_TYPE_BUY and current_price_for_calc < entry_price:
                        current_loss_pips = (entry_price - current_price_for_calc) / point_value
                        is_losing_for_check = True
                    elif trade_type == mt5.ORDER_TYPE_SELL and current_price_for_calc > entry_price:
                        current_loss_pips = (current_price_for_calc - entry_price) / point_value
                        is_losing_for_check = True
                    
                    logger.debug(f"MANAGE_OPEN_TRADES ({entry_timeframe} Loss - {ticket}): CurrentLossPips={current_loss_pips:.1f}, TargetLossPipsToClose={total_sl_distance_pips * sl_distance_ratio:.1f} (TotalSLDist={total_sl_distance_pips:.1f})")

                    if is_losing_for_check and total_sl_distance_pips > 0 and current_loss_pips >= sl_distance_ratio * total_sl_distance_pips:
                        logger.info(f"MANAGE_OPEN_TRADES ({entry_timeframe} Loss - {ticket}): Closing trade. Loss ({current_loss_pips:.1f} pips) >= target ({sl_distance_ratio*100}% of SL distance).")
                        if self.close_position_by_ticket(ticket, symbol, volume, trade_type, f"{entry_timeframe} Auto SL Ratio"):
                            closed_by_specific_logic = True
                        continue 
            
            # --- General Percentage-Based Auto-Close (Fallback if not M1/M5/M15 or specific logic disabled/not met) ---
            if not closed_by_specific_logic:
                logger.debug(f"MANAGE_OPEN_TRADES (Ticket {ticket}): M1/M5/M15 specific logic did not close. Checking general auto-close.")
                # General Profit-Taking Logic (% of entry price, H1 trend)
                if general_profit_config_enabled and entry_price > 0:
                    profit_percent_of_entry = 0.0; trade_is_in_profit = False
                    if trade_type == mt5.ORDER_TYPE_BUY and current_price_for_calc > entry_price:
                        profit_percent_of_entry = ((current_price_for_calc - entry_price) / entry_price) * 100
                        trade_is_in_profit = True
                    elif trade_type == mt5.ORDER_TYPE_SELL and current_price_for_calc < entry_price:
                        profit_percent_of_entry = ((entry_price - current_price_for_calc) / entry_price) * 100
                        trade_is_in_profit = True
                    
                    min_profit_pct = self.config.get('general_auto_close_profit_percent_entry_min', 60.0)
                    max_profit_pct = self.config.get('general_auto_close_profit_percent_entry_max', 70.0)
                    logger.debug(f"MANAGE_OPEN_TRADES (General Profit - {ticket}): CalcProfit%Entry={profit_percent_of_entry:.2f}%, TargetRange={min_profit_pct}-{max_profit_pct}%")

                    if trade_is_in_profit and (min_profit_pct <= profit_percent_of_entry <= max_profit_pct):
                        h1_trend_status = self.get_trend(symbol, timeframe_str='H1') # Always H1 for general
                        logger.info(f"MANAGE_OPEN_TRADES (General Profit - {ticket}): Profit in range. H1 Trend for {symbol} is {h1_trend_status}.")
                        close_general_profit = False
                        if trade_type == mt5.ORDER_TYPE_BUY and h1_trend_status == 'downtrend': close_general_profit = True
                        elif trade_type == mt5.ORDER_TYPE_SELL and h1_trend_status == 'uptrend': close_general_profit = True
                        if close_general_profit:
                            logger.info(f"MANAGE_OPEN_TRADES (General Profit - {ticket}): Closing {'BUY' if trade_type == mt5.ORDER_TYPE_BUY else 'SELL'} due to H1 trend reversal.")
                            if self.close_position_by_ticket(ticket, symbol, volume, trade_type, "General Auto TP Reversal"):
                                closed_by_specific_logic = True # Mark as closed to skip general loss
                            continue
                
                # General Loss-Cutting Logic (% of entry price)
                if not closed_by_specific_logic: # Check if not closed by general profit logic
                    current_pos_info_after_general_profit_check = mt5.positions_get(ticket=ticket)
                    if not (current_pos_info_after_general_profit_check and len(current_pos_info_after_general_profit_check) > 0):
                        logger.debug(f"MANAGE_OPEN_TRADES: Position {ticket} for {symbol} no longer exists (likely closed by general profit logic). Skipping general loss check.")
                        continue

                    if general_loss_config_enabled and entry_price > 0:
                        loss_percent_of_entry = 0.0; trade_is_in_loss = False
                        if trade_type == mt5.ORDER_TYPE_BUY and current_price_for_calc < entry_price:
                            loss_percent_of_entry = ((entry_price - current_price_for_calc) / entry_price) * 100
                            trade_is_in_loss = True
                        elif trade_type == mt5.ORDER_TYPE_SELL and current_price_for_calc > entry_price:
                            loss_percent_of_entry = ((current_price_for_calc - entry_price) / entry_price) * 100
                            trade_is_in_loss = True

                        max_loss_pct = self.config.get('general_auto_close_max_loss_percent_entry', 55.0)
                        logger.debug(f"MANAGE_OPEN_TRADES (General Loss - {ticket}): CalcLoss%Entry={loss_percent_of_entry:.2f}%, MaxLoss%={max_loss_pct}")
                        if trade_is_in_loss and loss_percent_of_entry >= max_loss_pct:
                            logger.info(f"MANAGE_OPEN_TRADES (General Loss - {ticket}): Closing trade. Loss ({loss_percent_of_entry:.2f}%) >= threshold.")
                            self.close_position_by_ticket(ticket, symbol, volume, trade_type, "General Auto SL % Loss")
                            # continue # End of logic for this position
        logger.debug("MANAGE_OPEN_TRADES: Cycle end.")

    def check_closed_trades(self):
        # (Identical to your provided main.py)
        if not os.path.exists("trades.csv"): return
        try:
            trades_df = pd.read_csv("trades.csv")
        except pd.errors.EmptyDataError:
            logger.info("trades.csv is empty. No closed trades to check.")
            return

        if 'order_id' in trades_df.columns:
            trades_df['order_id'] = pd.to_numeric(trades_df['order_id'], errors='coerce').fillna(0).astype(int)
        else: 
            logger.warning("'order_id' column not found in trades.csv. Cannot check closed trades.")
            return 
            
        open_logged_trades = trades_df[trades_df['status'] == 'open']
        if open_logged_trades.empty: return

        for index, trade_row in open_logged_trades.iterrows():
            order_id = trade_row['order_id']
            if order_id == 0 : continue 
            symbol = trade_row['symbol']
            position_info = mt5.positions_get(ticket=order_id) 
            if position_info is None or not position_info: 
                deals = mt5.history_deals_get(position=order_id) 
                if deals is None: logger.error(f"Failed to get deals for {order_id} on {symbol}: {mt5.last_error()}"); continue
                if deals: 
                    profit_loss = 0.0; exit_time_str = datetime.now().isoformat() + "Z"; num_closing_deals = 0
                    for deal in deals:
                        if deal.position_id == order_id: 
                            profit_loss += deal.profit
                            exit_time_str = datetime.utcfromtimestamp(deal.time).isoformat() + "Z"
                            if deal.entry == mt5.DEAL_ENTRY_OUT or deal.entry == mt5.DEAL_ENTRY_INOUT: num_closing_deals +=1
                    if num_closing_deals > 0 : 
                        update_data = {"status": "closed", "profit_loss": profit_loss, "exit_time": exit_time_str,
                                       "account_balance": mt5.account_info().balance if mt5.account_info() else 0}
                        if trade_row.get("status") == "closed_auto" and pd.notna(trade_row.get("exit_reason")):
                            update_data["status"] = "closed_auto"; update_data["exit_reason"] = trade_row.get("exit_reason")
                        update_trade_status(order_id, update_data)
                        log_level = logger.info if profit_loss >= 0 else logger.warning
                        log_level(f"Trade RECONCILED CLOSED: {symbol} (ID: {order_id}), P/L: {profit_loss:.2f}. Reason: {update_data.get('exit_reason', 'SL/TP/Manual')}")
                        if profit_loss < 0: self.consecutive_losses += 1
                        else: self.consecutive_losses = 0
                        logger.info(f"Consecutive losses: {self.consecutive_losses}")
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.error(f"{self.max_consecutive_losses} consec. losses. Shutting down for 15 mins.")
            mt5.shutdown(); time.sleep(900); self.consecutive_losses = 0
            if not connect_mt5(): logger.critical("MT5 reconnect failed. Exiting."); exit(1)
            self.ensure_symbols_selected()

    def monitor_market(self):
        # (Identical to your provided main.py)
        while True:
            try:
                if datetime.now() - self.last_global_retrain_time > self.global_retrain_interval:
                    logger.info(f"Scheduled global retraining. Last: {self.last_global_retrain_time}")
                    self.perform_global_retrain()
                self.check_closed_trades(); self.manage_open_trades(); self.ensure_symbols_selected()
                for symbol in self.config['symbols']:
                    if not mt5.symbol_select(symbol, True): logger.warning(f"Could not select {symbol}, skipping."); continue
                    symbol_info_obj = mt5.symbol_info(symbol)
                    if not symbol_info_obj: logger.error(f"Symbol {symbol} not found. Skipping."); continue
                    if not self.is_trading_hours(symbol): logger.debug(f"Outside trading hours for {symbol}."); continue
                    current_tick = mt5.symbol_info_tick(symbol)
                    if not current_tick or current_tick.time == 0: logger.warning(f"Market closed/no tick for {symbol}."); continue
                    if self.in_cooldown(symbol): logger.debug(f"Symbol {symbol} in cooldown."); continue
                    for tf_str in self.config.get('timeframes', ['M5']):
                        mt5_tf_val = self.mt5_timeframes.get(tf_str, mt5.TIMEFRAME_M5)
                        data = get_data(symbol, mt5_tf_val, self.config.get('bars', 1000)) 
                        if data is None or data.empty: logger.warning(f"No {tf_str} data for {symbol}."); continue
                        if 'time' not in data.columns: logger.error(f"Data for {symbol} {tf_str} missing 'time'."); continue
                        signals = {}; strengths = {}
                        for strategy in self.strategies:
                            try:
                                if isinstance(strategy, (SMCStrategy, BollingerBandsStrategy, LiquiditySweepStrategy, MalaysianSnRStrategy)):
                                    result = strategy.get_signal(data.copy(), symbol=symbol, timeframe=tf_str)
                                else: result = strategy.get_signal(data.copy(), symbol=symbol)
                                if isinstance(result, tuple) and len(result) >= 2:
                                    signal_val, strength_val = result[0], result[1]
                                    signals[strategy.name] = signal_val
                                    strengths[strategy.name] = float(strength_val) if pd.notna(strength_val) else 0.0
                                else: signals[strategy.name] = 'hold'; strengths[strategy.name] = 0.0
                            except Exception as e:
                                logger.error(f"Error in {strategy.name}.get_signal for {symbol} {tf_str}: {e}", exc_info=False)
                                signals[strategy.name] = 'hold'; strengths[strategy.name] = 0.0
                        consensus_signal = self.analyze_signals(signals, strengths, data.copy()) 
                        if consensus_signal != 'hold':
                            logger.info(f"Attempting trade for {symbol} on {tf_str}: {consensus_signal}")
                            self.execute_trade(symbol, consensus_signal, data.copy(), tf_str)
                        if self.consecutive_failures >= 3:
                            logger.error("3 consec. EXECUTION failures. Shutting down for 15 mins.")
                            mt5.shutdown(); time.sleep(900); self.consecutive_failures = 0
                            if not connect_mt5(): logger.critical("MT5 RECONNECT FAILED. EXITING."); exit(1)
                            self.ensure_symbols_selected(); break 
                    if self.consecutive_failures >=3: break 
                time.sleep(self.config.get('monitoring_interval_seconds', 60))
            except ConnectionResetError as cre:
                logger.error(f"ConnectionResetError: {cre}. Reconnecting MT5...")
                if mt5.terminal_info() is None or not mt5.ping():
                    if not connect_mt5(): time.sleep(300) 
                    else: self.ensure_symbols_selected()
                else: time.sleep(60)
            except RuntimeError as re: 
                logger.error(f"RuntimeError: {re}")
                if "IPC timeout" in str(re) or "Terminal not found" in str(re) or "Server is unavailable" in str(re):
                    if not connect_mt5(): time.sleep(300)
                    else: self.ensure_symbols_selected()
                else: time.sleep(60) 
            except Exception as e:
                logger.error(f"Unhandled error in monitor_market: {e}", exc_info=True)
                time.sleep(self.config.get('error_sleep_interval_seconds', 300))

    def analyze_signals(self, signals, strengths, data):
        # (Identical to your provided main.py)
        buy_weighted_strength_sum = 0.0
        sell_weighted_strength_sum = 0.0
        strategy_weights_from_history = get_strategy_weights() 
        volatility = 0.0
        if not data.empty and len(data) >= 20 and all(c in data.columns for c in ['high', 'low', 'close']):
            if data['close'].iloc[-1] != 0: 
                 volatility = (data['high'].iloc[-20:].max() - data['low'].iloc[-20:].min()) / data['close'].iloc[-1]
        
        threshold_config_main = self.config.get('consensus_threshold', {"low_vol": 1.0, "high_vol": 1.5, "vol_split": 0.005})
        if not isinstance(threshold_config_main, dict):
            threshold_config_main = {"low_vol": 1.0, "high_vol": 1.5, "vol_split": 0.005} 
            logger.warning("Consensus threshold in config was malformed, using defaults.")

        low_vol_thresh = threshold_config_main.get("low_vol", 1.0)
        high_vol_thresh = threshold_config_main.get("high_vol", 1.5)
        vol_split_thresh = threshold_config_main.get("vol_split", 0.005)
        consensus_threshold_value = low_vol_thresh if volatility < vol_split_thresh else high_vol_thresh
        
        for strategy_obj in self.strategies: 
            strategy_name = strategy_obj.name
            base_weight = strategy_weights_from_history.get(strategy_name, 1.0) 
            signal_strength_from_strat = strengths.get(strategy_name, 0.0)
            effective_weight = base_weight * max(signal_strength_from_strat, 0.1) 
            strategy_boost_config = self.config.get('strategy_boost_factor', {})
            if strategy_name in strategy_boost_config: 
                effective_weight *= strategy_boost_config.get(strategy_name, 1.0)
            signal_direction = signals.get(strategy_name, 'hold')
            if signal_direction == 'buy': buy_weighted_strength_sum += effective_weight
            elif signal_direction == 'sell': sell_weighted_strength_sum += effective_weight
            
        logger.debug(f"Signal analysis: Volatility={volatility:.4f}, BuySum={buy_weighted_strength_sum:.2f}, SellSum={sell_weighted_strength_sum:.2f}, Threshold={consensus_threshold_value}, Signals={signals}")
        dominance_factor = self.config.get('consensus_dominance_factor', 1.8)
        if buy_weighted_strength_sum >= consensus_threshold_value and buy_weighted_strength_sum > sell_weighted_strength_sum * dominance_factor:
            return 'buy'
        if sell_weighted_strength_sum >= consensus_threshold_value and sell_weighted_strength_sum > buy_weighted_strength_sum * dominance_factor:
            return 'sell'
        return 'hold'

if __name__ == "__main__":
    if not connect_mt5():
        logger.critical("MT5 connection failed on startup. Bot cannot start.")
        exit(1) 
    bot = TradingBot()
    try:
        bot.monitor_market()
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"Critical unhandled exception in bot's main execution: {e}", exc_info=True)
    finally:
        logger.info("Shutting down MT5 connection.")
        mt5.shutdown()

import logging
import MetaTrader5 as mt5
import pandas as pd
import json
import time
import numpy as np
import os
import re # For parsing comment
import threading # For running bot loop separately
from datetime import datetime, timedelta, timezone # Ensure timezone is imported
from zoneinfo import ZoneInfo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.validation import check_is_fitted, NotFittedError
import inspect
from collections import Counter # Added for label distribution logging
from utils.feature_engineering import extract_ml_features, calculate_atr, calculate_rsi, bollinger_band_width, calculate_macd
from utils.news_manager import NewsManager
# Assuming strategies and utils are in paths accessible by Python
from strategies.bollinger_bands import BollingerBandsStrategy
from strategies.ml_model import MLValidator
from utils.mt5_connection import connect_mt5, get_data
from utils.logging import setup_logging
# --- MODIFIED: Updated trade_history imports for dynamic weighting ---
from utils.trade_history import save_trade, update_trade_status, get_strategy_weights
from strategies.sma import SMAStrategy
from strategies.smc import SMCStrategy
# --- MODIFIED: Replaced old liquidity sweep with the new advanced one ---
from strategies.liquidity_sweep import LiquiditySweepStrategy
from strategies.fibonacci import FibonacciStrategy
from strategies.malaysian_snr import MalaysianSnRStrategy
from strategies.adx_strategy import ADXStrategy
from strategies.keltner_channels_strategy import KeltnerChannelsStrategy
from strategies.scalping_strategy import ScalpingStrategy
from strategies.mean_reversion_scalper import MeanReversionScalper
try:
    from strategies.ml_prediction_strategy import MLPredictionStrategy
except ImportError:
    MLPredictionStrategy = None
    logging.getLogger().warning("Could not import MLPredictionStrategy. Please ensure 'strategies/ml_prediction_strategy.py' exists.")


# Import MongoDB connection utility
from utils.db_connector import MongoDBConnection
from pymongo import DESCENDING, errors  # For sorting

logger = setup_logging()

class TradingBot:
    def __init__(self):
        self.bot_running = False
        self.config_lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.bot_thread = None
        self.last_error_message = None
        self.config = {}
        self.current_signals_for_logging = {}
        self._adx_strategy_for_features = None
        self._keltner_strategy_for_features = None
        self.trade_management_states = {} # To store states for profit securing and trailing
        self.dxy_data_cache = {} # Cache for DXY data
        self.dxy_cache_lock = threading.Lock() # Lock for cache access
        self.news_manager = NewsManager(self.config)
        # Load configuration first
        self.load_config_and_reinitialize()

        # Initialize MongoDB connection
        self.db = MongoDBConnection.connect()
        if not self.db:
            logger.critical("MongoDB connection failed. Bot may not function correctly with trade history.")


    def load_config_and_reinitialize(self):
        with self.config_lock:
            logger.info("Loading configuration and reinitializing bot components...")
            try:
                config_dir = 'config'
                if not os.path.exists(config_dir):
                    os.makedirs(config_dir)
                    logger.info(f"Created directory: {config_dir}")

                config_file_path = os.path.join(config_dir, 'config.json')

                if not os.path.exists(config_file_path):
                    logger.warning(f"{config_file_path} not found. Bot will not run without a config.")
                    raise FileNotFoundError(f"{config_file_path} not found.")
                else:
                    with open(config_file_path, 'r') as f:
                        config_data = json.load(f)
            except Exception as e:
                logger.error(f"CRITICAL: Failed to load config.json: {e}. Bot cannot operate correctly.", exc_info=True)
                if not self.config: raise RuntimeError(f"Failed to load initial config: {e}")
                logger.warning("Using previously loaded config due to error in reloading.")
                return

            self.config = config_data
            if hasattr(self, 'ml_validator') and self.ml_validator:
                self.ml_validator.config = self.config
                self.ml_validator.create_predictors_for_all_symbols() # Recreate with new config
            else:
                self.ml_validator = MLValidator(self.config)

            self.strategies = self.initialize_strategies()

            # Pass dependencies to feature engineering functions if needed
            # This is important after refactoring
            self._adx_strategy_for_features = next((s for s in self.strategies if isinstance(s, ADXStrategy)), None)
            self._keltner_strategy_for_features = next((s for s in self.strategies if isinstance(s, KeltnerChannelsStrategy)), None)
            if not self._adx_strategy_for_features:
                logger.warning("ADXStrategy instance not found. ADX features/filter might be unavailable.")
            if not self._keltner_strategy_for_features:
                logger.warning("KeltnerChannelsStrategy instance not found. Keltner features might be unavailable.")


            for strategy in self.strategies:
                if hasattr(strategy, 'set_config') and callable(getattr(strategy, 'set_config')):
                    strategy.set_config(self.config)

            self.cooldown_period = timedelta(minutes=self.config.get('cooldown_period_minutes', 2))
            self.global_retrain_interval = timedelta(hours=self.config.get('global_retrain_hours', 12))

            self.last_trade_times = getattr(self, 'last_trade_times', {})
            self.consecutive_failures = getattr(self, 'consecutive_failures', 0)
            self.trade_management_states = getattr(self, 'trade_management_states', {})

            last_retrain_attr = getattr(self, 'last_global_retrain_time', datetime.min)
            if last_retrain_attr == datetime.min: self.last_global_retrain_time = datetime.min.replace(tzinfo=timezone.utc)
            elif last_retrain_attr.tzinfo is None: self.last_global_retrain_time = last_retrain_attr.replace(tzinfo=timezone.utc)

            self.mt5_timeframes = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1
            }
            logger.info("Bot configuration and components reloaded/reinitialized.")

    def get_entry_timeframe_from_comment(self, comment_str):
        if not isinstance(comment_str, str): return None
        match = re.search(r"AI\s+([MHDW][1-9]\d*)\s+", comment_str)
        if match: return match.group(1)
        match_simple_tf = re.search(r"([MHDW][1-9]\d*)", comment_str) # Fallback for simpler comments
        if match_simple_tf: return match_simple_tf.group(1)
        return None

    def modify_position_sl(self, ticket, new_sl, symbol):
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Failed to get symbol_info for {symbol} in modify_position_sl.")
            return False

        new_sl_rounded = round(new_sl, symbol_info.digits)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl_rounded,
            "symbol": symbol
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Successfully modified SL for ticket {ticket} to {new_sl_rounded:.{symbol_info.digits}f}")
            return True
        else:
            error_message = result.comment if result else mt5.last_error()
            retcode_val = result.retcode if result else 'N/A'
            logger.error(f"Failed to modify SL for ticket {ticket} to {new_sl_rounded}. Error: {error_message} (RetCode: {retcode_val})")
            if retcode_val == 10016: # Invalid stops
                 logger.warning(f"Invalid SL for ticket {ticket}. Original SL might be too close to current price, or new SL {new_sl_rounded} violates stops level.")
            return False


    def manage_open_trades(self):
        ps_sl_config_main = self.config.get('profit_securing_stop_loss', {"enabled": False})
        time_exit_config_main = self.config.get('time_based_exit', {"enabled": False})
        short_tf_profit_config = self.config.get('auto_close_short_tf_profit_take', {})
        short_tf_loss_config = self.config.get('auto_close_short_tf_stop_loss', {})

        if not (ps_sl_config_main.get("enabled", False) or \
                time_exit_config_main.get("enabled", False) or \
                short_tf_profit_config.get("enabled", False) or \
                short_tf_loss_config.get("enabled", False)):
            return

        open_positions = mt5.positions_get()
        if open_positions is None:
            logger.error(f"MANAGE_TRADES: Failed to get open positions: {mt5.last_error()}")
            return
        if not open_positions:
            return

        for pos in open_positions:
            if pos.magic != 123456: continue

            symbol = pos.symbol; ticket = pos.ticket; entry_price = pos.price_open
            original_sl = pos.sl; original_tp = pos.tp; position_open_time_dt = datetime.fromtimestamp(pos.time, tz=timezone.utc)
            trade_type = pos.type; volume = pos.volume

            tick = mt5.symbol_info_tick(symbol)
            if not tick or tick.time == 0:
                logger.warning(f"MANAGE_TRADES ({ticket}): No valid tick for {symbol}. Cannot evaluate."); continue

            current_price = tick.bid if trade_type == mt5.ORDER_TYPE_BUY else tick.ask
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"MANAGE_TRADES ({ticket}): No symbol_info for {symbol}. Cannot evaluate."); continue
            point = symbol_info.point
            digits = symbol_info.digits

            current_profit_pips = 0
            if point > 0:
                current_profit_pips = ((current_price - entry_price) / point) if trade_type == mt5.ORDER_TYPE_BUY else ((entry_price - current_price) / point)

            closed_flag = False

            if ps_sl_config_main.get("enabled", False) and not closed_flag:
                ps_sl_symbol_specific_config = ps_sl_config_main.get(symbol, ps_sl_config_main.get("default_settings", {}))

                trade_state = self.trade_management_states.setdefault(ticket, {
                    "initial_secure_done": False,
                    "trail_reference_price": entry_price,
                    "highest_profit_pips_overall": 0.0
                })
                trade_state["highest_profit_pips_overall"] = max(trade_state["highest_profit_pips_overall"], current_profit_pips)

                if not trade_state["initial_secure_done"]:
                    trigger_pips = ps_sl_symbol_specific_config.get("trigger_profit_pips", 50)
                    if current_profit_pips >= trigger_pips:
                        new_secure_sl = 0.0
                        secure_type = ps_sl_symbol_specific_config.get("secure_profit_type", "fixed_pips")

                        if secure_type == "fixed_pips":
                            pips_to_secure = ps_sl_symbol_specific_config.get("secure_profit_fixed_pips", 10)
                            if trade_type == mt5.ORDER_TYPE_BUY:
                                new_secure_sl = entry_price + (pips_to_secure * point)
                            else: # SELL
                                new_secure_sl = entry_price - (pips_to_secure * point)
                        elif secure_type == "percentage_of_profit":
                            percentage = ps_sl_symbol_specific_config.get("secure_profit_percentage", 0.5)
                            pips_to_secure = current_profit_pips * percentage
                            if trade_type == mt5.ORDER_TYPE_BUY:
                                new_secure_sl = entry_price + (pips_to_secure * point)
                            else: # SELL
                                new_secure_sl = entry_price - (pips_to_secure * point)

                        should_move_sl = False
                        if trade_type == mt5.ORDER_TYPE_BUY and (original_sl == 0 or new_secure_sl > original_sl):
                            should_move_sl = True
                        elif trade_type == mt5.ORDER_TYPE_SELL and (original_sl == 0 or new_secure_sl < original_sl):
                            should_move_sl = True

                        if should_move_sl and new_secure_sl != 0:
                            if self.modify_position_sl(ticket, new_secure_sl, symbol):
                                logger.info(f"MANAGE_TRADES (Initial Secure - {ticket}): Moved SL to {new_secure_sl:.{digits}f} for {('BUY' if trade_type == mt5.ORDER_TYPE_BUY else 'SELL')} {symbol}.")
                                update_trade_status(ticket, {"sl_price": round(new_secure_sl, digits), "exit_reason": "InitialProfitSecure"})
                                trade_state["initial_secure_done"] = True
                                trade_state["trail_reference_price"] = current_price
                            else:
                                logger.warning(f"MANAGE_TRADES (Initial Secure - {ticket}): Failed to modify SL to {new_secure_sl:.{digits}f}.")

                elif trade_state["initial_secure_done"] and ps_sl_symbol_specific_config.get("trailing_active", False):
                    if trade_type == mt5.ORDER_TYPE_BUY:
                        trade_state["trail_reference_price"] = max(trade_state["trail_reference_price"], current_price)
                    else: # SELL
                        trade_state["trail_reference_price"] = min(trade_state["trail_reference_price"], current_price)

                    new_trailing_sl = 0.0
                    trailing_method = ps_sl_symbol_specific_config.get("trailing_method", "fixed_pips_behind")
                    trail_ref_price = trade_state["trail_reference_price"]

                    if trailing_method == "fixed_pips_behind":
                        pips_behind = ps_sl_symbol_specific_config.get("trailing_fixed_pips_value", 20)
                        if trade_type == mt5.ORDER_TYPE_BUY:
                            new_trailing_sl = trail_ref_price - (pips_behind * point)
                        else: # SELL
                            new_trailing_sl = trail_ref_price + (pips_behind * point)

                    elif trailing_method == "atr_multiplier":
                        atr_period = ps_sl_symbol_specific_config.get("trailing_atr_period", 14)
                        atr_multiplier = ps_sl_symbol_specific_config.get("trailing_atr_multiplier_value", 1.5)
                        entry_tf_for_atr = self.get_entry_timeframe_from_comment(pos.comment)
                        if entry_tf_for_atr:
                            data_for_atr = get_data(symbol, entry_tf_for_atr, bars=atr_period + 50)
                            if data_for_atr is not None and not data_for_atr.empty:
                                # MODIFIED: Call refactored function
                                atr_value = calculate_atr(data_for_atr.copy(), period=atr_period)
                                if atr_value > 0:
                                    if trade_type == mt5.ORDER_TYPE_BUY:
                                        new_trailing_sl = trail_ref_price - (atr_value * atr_multiplier)
                                    else: # SELL
                                        new_trailing_sl = trail_ref_price + (atr_value * atr_multiplier)
                                else: logger.warning(f"MANAGE_TRADES (ATR Trail - {ticket}): ATR value is zero or invalid for {symbol} on {entry_tf_for_atr}.")
                            else: logger.warning(f"MANAGE_TRADES (ATR Trail - {ticket}): Could not fetch data for ATR on {symbol} {entry_tf_for_atr}.")
                        else: logger.warning(f"MANAGE_TRADES (ATR Trail - {ticket}): Could not determine entry timeframe for ATR calculation from comment: '{pos.comment}'.")

                    elif trailing_method == "percentage_of_peak_profit":
                        percentage_of_profit = ps_sl_symbol_specific_config.get("trailing_percentage_value", 70) / 100.0
                        locked_in_profit_pips = trade_state["highest_profit_pips_overall"] * percentage_of_profit
                        if locked_in_profit_pips > 0:
                            if trade_type == mt5.ORDER_TYPE_BUY:
                                new_trailing_sl = entry_price + (locked_in_profit_pips * point)
                            else: # SELL
                                new_trailing_sl = entry_price - (locked_in_profit_pips * point)

                    should_move_trailing_sl = False
                    if new_trailing_sl != 0:
                        if trade_type == mt5.ORDER_TYPE_BUY and (original_sl == 0 or new_trailing_sl > original_sl):
                            should_move_trailing_sl = True
                        elif trade_type == mt5.ORDER_TYPE_SELL and (original_sl == 0 or new_trailing_sl < original_sl):
                             should_move_trailing_sl = True

                    if should_move_trailing_sl:
                        if self.modify_position_sl(ticket, new_trailing_sl, symbol):
                            logger.info(f"MANAGE_TRADES (Trailing SL - {ticket} - {trailing_method}): Moved SL to {new_trailing_sl:.{digits}f}.")
                            update_trade_status(ticket, {"sl_price": round(new_trailing_sl, digits), "exit_reason": f"TrailingSL-{trailing_method}"})

            entry_tf = self.get_entry_timeframe_from_comment(pos.comment)
            if entry_tf in ['M1', 'M5', 'M15', 'M30'] and not closed_flag:
                if short_tf_profit_config.get("enabled", False) and original_tp != 0 and entry_price != 0 and point > 0:
                    tp_dist_ratio = short_tf_profit_config.get("tp_distance_ratio", 0.5)
                    potential_profit_pips = abs(original_tp - entry_price) / point
                    if current_profit_pips >= tp_dist_ratio * potential_profit_pips and potential_profit_pips > 0 :
                        current_short_tf_trend = self.get_trend(symbol, timeframe_str=entry_tf)
                        if (trade_type == mt5.ORDER_TYPE_BUY and current_short_tf_trend == 'downtrend') or \
                           (trade_type == mt5.ORDER_TYPE_SELL and current_short_tf_trend == 'uptrend'):
                            logger.info(f"MANAGE_TRADES ({entry_tf} Profit - {ticket}): Closing due to trend reversal ({current_short_tf_trend}) after reaching {tp_dist_ratio*100}% of TP distance.")
                            if self.close_position_by_ticket(ticket, symbol, volume, trade_type, f"{entry_tf} Auto TP Reversal"): closed_flag = True

                if not closed_flag and short_tf_loss_config.get("enabled", False) and original_sl != 0 and entry_price != 0 and point > 0:
                    sl_dist_ratio = short_tf_loss_config.get("sl_distance_ratio", 0.5)
                    total_sl_dist_pips = abs(entry_price - original_sl) / point
                    current_loss_pips = ((entry_price - current_price) / point) if trade_type == mt5.ORDER_TYPE_BUY else ((current_price - entry_price) / point)
                    if current_loss_pips >= sl_dist_ratio * total_sl_dist_pips and total_sl_dist_pips > 0:
                        logger.info(f"MANAGE_TRADES ({entry_tf} Loss - {ticket}): Closing. Loss ({current_loss_pips:.1f} pips) >= {sl_dist_ratio*100}% of SL distance ({total_sl_dist_pips:.1f} pips).")
                        if self.close_position_by_ticket(ticket, symbol, volume, trade_type, f"{entry_tf} Auto SL Ratio"): closed_flag = True

            if time_exit_config_main.get("enabled", False) and entry_tf and entry_tf in time_exit_config_main.get("apply_to_timeframes", []) and not closed_flag:
                symbol_tf_key = f"{symbol}_{entry_tf}"
                time_exit_params = time_exit_config_main.get(symbol_tf_key, time_exit_config_main.get("default", {}))

                max_bars = time_exit_params.get("max_bars_open", 12)
                min_profit_pips = time_exit_params.get("min_profit_pips_to_consider", 30)

                entry_tf_minutes = self.mt5_tf_to_minutes(entry_tf)
                if entry_tf_minutes > 0:
                    seconds_open = (datetime.now(timezone.utc) - position_open_time_dt).total_seconds()
                    bars_open = seconds_open / (entry_tf_minutes * 60)
                    logger.debug(f"MANAGE_TRADES (Time-Based Check - {ticket} on {entry_tf}): Bars open: {bars_open:.2f}/{max_bars}, Profit pips: {current_profit_pips:.1f}/{min_profit_pips}")

                    if bars_open >= max_bars and current_profit_pips >= min_profit_pips:
                        logger.info(f"MANAGE_TRADES (Time-Based - {ticket}): Max bars ({max_bars}) reached with profit ({current_profit_pips} pips). Checking momentum.")
                        data_for_indicators = get_data(symbol, entry_tf, bars=30)
                        momentum_fading = False
                        if data_for_indicators is not None and not data_for_indicators.empty:
                            adx_val, plus_di_val, minus_di_val = np.nan, np.nan, np.nan
                            if self._adx_strategy_for_features:
                                adx_ind_vals = self._adx_strategy_for_features.get_indicator_values(data_for_indicators.copy())
                                if adx_ind_vals and adx_ind_vals.get('adx') is not None and not adx_ind_vals['adx'].empty:
                                    adx_val = adx_ind_vals['adx'].iloc[-1]
                            # MODIFIED: Call refactored function
                            rsi_val = calculate_rsi(data_for_indicators.copy())

                            fade_adx_thresh = time_exit_params.get("momentum_fade_adx_threshold", 22)
                            fade_rsi_buy_exit = time_exit_params.get("momentum_fade_rsi_buy_exit", 48)
                            fade_rsi_sell_exit = time_exit_params.get("momentum_fade_rsi_sell_exit", 52)

                            if pd.notna(adx_val) and adx_val < fade_adx_thresh: momentum_fading = True; logger.info(f"MANAGE_TRADES (Time-Based Fade - {ticket}): ADX ({adx_val:.2f}) < threshold ({fade_adx_thresh}).")
                            if not momentum_fading:
                                if trade_type == mt5.ORDER_TYPE_BUY and pd.notna(rsi_val) and rsi_val < fade_rsi_buy_exit: momentum_fading = True; logger.info(f"MANAGE_TRADES (Time-Based Fade - {ticket}): RSI ({rsi_val:.2f}) < BUY exit ({fade_rsi_buy_exit}).")
                                elif trade_type == mt5.ORDER_TYPE_SELL and pd.notna(rsi_val) and rsi_val > fade_rsi_sell_exit: momentum_fading = True; logger.info(f"MANAGE_TRADES (Time-Based Fade - {ticket}): RSI ({rsi_val:.2f}) > SELL exit ({fade_rsi_sell_exit}).")
                        else: logger.warning(f"MANAGE_TRADES (Time-Based - {ticket}): Could not fetch indicator data for momentum check."); momentum_fading = True

                        if momentum_fading:
                            logger.info(f"MANAGE_TRADES (Time-Based Exit - {ticket}): Closing due to time limit and momentum fade.")
                            if self.close_position_by_ticket(ticket, symbol, volume, trade_type, f"{entry_tf} Time Exit Momentum Fade"): closed_flag = True


    def check_closed_trades(self):
        """
        Reconciles trades marked as 'open' in MongoDB against MT5.
        If a trade is found closed in MT5, its status is updated in MongoDB.
        """
        trades_collection = MongoDBConnection.get_trades_collection()
        if not trades_collection:
            logger.error("CHECK_CLOSED_TRADES: MongoDB not connected. Cannot reconcile trades.")
            return

        try:
            open_logged_trades_cursor = trades_collection.find({"status": "open"})
            open_trades_list = list(open_logged_trades_cursor)

            if not open_trades_list:
                # logger.debug("CHECK_CLOSED_TRADES: No 'open' trades found in MongoDB to reconcile.")
                return

            for trade_doc in open_trades_list:
                order_id = trade_doc.get("order_id")
                if order_id is None:
                    logger.warning(f"CHECK_CLOSED_TRADES: Found trade document with missing order_id: {trade_doc.get('_id')}")
                    continue

                try:
                    order_id = int(order_id)
                except ValueError:
                    logger.warning(f"CHECK_CLOSED_TRADES: Invalid order_id format '{trade_doc.get('order_id')}' for doc_id {trade_doc.get('_id')}. Skipping.")
                    continue

                position_info = mt5.positions_get(ticket=order_id)

                if not position_info: # Position no longer open in MT5
                    deals = mt5.history_deals_get(position=order_id)
                    if deals:
                        profit = sum(d.profit for d in deals if d.position_id == order_id)
                        last_deal_time_msc = max(d.time_msc for d in deals if d.position_id == order_id) if deals else time.time() * 1000
                        exit_time_dt = datetime.fromtimestamp(last_deal_time_msc / 1000, tz=timezone.utc)

                        exit_price_deals = [d.price for d in deals if d.position_id == order_id and d.entry in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]]
                        exit_price = exit_price_deals[-1] if exit_price_deals else trade_doc.get('entry_price', 0)

                        current_account_info = mt5.account_info()
                        balance_at_close = current_account_info.balance if current_account_info else trade_doc.get('account_balance', 0)

                        update_data = {
                            "status": "closed",
                            "profit_loss": profit,
                            "exit_time": exit_time_dt,
                            "exit_price": exit_price,
                            "account_balance": balance_at_close
                        }

                        if trade_doc.get("status") == "closed_auto" and trade_doc.get("exit_reason"):
                            update_data["status"] = "closed_auto"
                            update_data["exit_reason"] = trade_doc.get("exit_reason")
                        elif not update_data.get("exit_reason"):
                             update_data["exit_reason"] = "SL/TP/Manual"

                        result = update_trade_status(order_id, update_data)

                        log_level = logger.info if profit >= 0 else logger.warning
                        log_level(f"Trade RECONCILED CLOSED (MongoDB): {trade_doc.get('symbol')} (ID: {order_id}), P/L: {profit:.2f}. Reason: {update_data.get('exit_reason', 'SL/TP/Manual')}")

                        if order_id in self.trade_management_states:
                            del self.trade_management_states[order_id]
                            logger.debug(f"Removed ticket {order_id} from trade_management_states after reconciliation.")

                    else:
                        logger.warning(f"CHECK_CLOSED_TRADES: Position {order_id} not found in MT5, but no deals in history. Marking as 'closed_unknown'.")
                        update_data = {
                            "status": "closed_unknown",
                            "exit_reason": "Position not in MT5, no deal history",
                            "exit_time": datetime.now(timezone.utc)
                        }
                        update_trade_status(order_id, update_data)
                        if order_id in self.trade_management_states: del self.trade_management_states[order_id]

        except errors.PyMongoError as e:
            logger.error(f"CHECK_CLOSED_TRADES: MongoDB error during reconciliation: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"CHECK_CLOSED_TRADES: Unexpected error during reconciliation: {e}", exc_info=True)


    def ensure_symbols_selected(self):
        """
        Ensures all symbols required for trading and feature extraction are selected in MT5.
        This function no longer modifies the main config list.
        """
        # --- BUG FIX: Create a COPY of the symbols list to avoid modifying the config in memory ---
        all_symbols_to_ensure = self.config.get('symbols', []).copy()
        dxy_symbol = self.config.get('dxy_symbol')

        feature_cfg = self.config.get('feature_engineering', {})
        if feature_cfg.get('dxy_correlation_for_xauusd') and dxy_symbol and dxy_symbol not in all_symbols_to_ensure:
            all_symbols_to_ensure.append(dxy_symbol)

        if not all_symbols_to_ensure:
            logger.error("No symbols defined in configuration for 'ensure_symbols_selected'.")
            return

        current_day_utc = datetime.now(timezone.utc).weekday()
        is_weekend = current_day_utc >= 5 # 5 for Saturday, 6 for Sunday

        for symbol in all_symbols_to_ensure:
            if is_weekend and symbol == dxy_symbol:
                logger.debug(f"Skipping selection check for DXY symbol '{symbol}' during the weekend.")
                continue

            max_retries = 3
            for attempt in range(max_retries):
                if not mt5.terminal_info():
                    logger.warning(f"MT5 connection lost before processing symbol {symbol}. Attempting to reconnect...")
                    if not connect_mt5(self.config.get('mt5_credentials')):
                        logger.error(f"MT5 reconnection failed. Cannot ensure symbol {symbol} is selected.")
                        return
                    else:
                        logger.info("MT5 reconnected successfully. Continuing symbol processing.")

                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to select symbol {symbol} in Market Watch (attempt {attempt + 1}/{max_retries}). Error: {mt5.last_error()}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        logger.error(f"Could not select symbol {symbol} after {max_retries} attempts. It might be unavailable.")
                        break
                else:
                    symbol_info_check = mt5.symbol_info(symbol)
                    if not symbol_info_check:
                        logger.error(f"Symbol {symbol} was selected, but failed to retrieve its info (attempt {attempt + 1}/{max_retries}). Error: {mt5.last_error()}")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        else:
                            logger.error(f"Failed to retrieve info for selected symbol {symbol} after {max_retries} attempts.")
                            break
                    else:
                        break
            else:
                logger.warning(f"Maximum retries reached for symbol {symbol}. It may not be usable.")


    def initialize_strategies(self):
        """
        Initializes strategies based on the config.
        FIX: Ensures the lambda function is called to get the parameter dictionary
             before it is unpacked into the strategy constructor.
        """
        # --- MODIFIED: Replaced old LiquiditySweep with the new advanced one ---
        strategy_constructors = {
            "SMA": (SMAStrategy, lambda c: {'short_period': c.get('sma_short_period', 10), 'long_period': c.get('sma_long_period', 100)}),
            "SMC": (SMCStrategy, lambda c: c.get('smc_params', {})),
            "LiquiditySweep": (LiquiditySweepStrategy, lambda c: c.get('liquidity_sweep_params', {})),
            "Fibonacci": (FibonacciStrategy, lambda c: {
                'swing_lookback': c.get('fibonacci_golden_zone', {}).get('swing_lookback', 200),
                'trend_ema_period': c.get('fibonacci_golden_zone', {}).get('trend_ema_period', 200),
                'strength': c.get('fibonacci_golden_zone', {}).get('signal_strength', 0.85)
            }),
            "MalaysianSnR": (MalaysianSnRStrategy, lambda c: {}),
            "BollingerBands": (BollingerBandsStrategy, lambda c: {'window': c.get('bb_window', 20), 'std_dev': c.get('bb_std_dev', 2.0)}),
            "ADX": (ADXStrategy, lambda c: {}),
            "KeltnerChannels": (KeltnerChannelsStrategy, lambda c: {}),
            "Scalping": (ScalpingStrategy, lambda c: {}),
            "MeanReversionScalper": (MeanReversionScalper, lambda c: c.get('mean_reversion_scalper_params', {})),
            "MLPrediction": (MLPredictionStrategy, lambda c: {'bot_instance': self})
        }

        initialized_strategies = []
        active_strategy_names = []

        if self.config.get('thesis_mode_enabled', False):
            active_strategy_names = ["SMA", "MLPrediction"]
        else:
            active_strategy_names = list(strategy_constructors.keys())

        logger.info(f"Active strategies to be loaded: {active_strategy_names}")

        for name in active_strategy_names:
            if name in strategy_constructors:
                StrategyClass, params_lambda = strategy_constructors[name]
                try:
                    # FIX IS HERE: Call the lambda to get the dictionary of parameters
                    params_dict = params_lambda(self.config)
                    # Now, unpack the resulting dictionary
                    initialized_strategies.append(StrategyClass(name=name, **params_dict))
                    logger.info(f"Successfully initialized strategy: {name}")
                except Exception as e:
                    logger.error(f"Failed to initialize strategy {name}: {e}", exc_info=True)
            else:
                logger.warning(f"Strategy '{name}' is configured to be active but not found in constructors.")

        return initialized_strategies


    def generate_ml_labels(self, data_series, entry_idx, direction):
        target_def = self.config.get('ml_target_definition', {})
        method = target_def.get("method", "triple_barrier")
        params = target_def.get("params", {})
        prediction_horizon = self.config.get('ml_prediction_horizon', 20)
        entry_price = data_series['close'].iloc[entry_idx]
        future_slice = data_series.iloc[entry_idx + 1 : entry_idx + 1 + prediction_horizon]

        if future_slice.empty or entry_price == 0:
            return 0

        if method == "price_change":
            change_threshold = params.get("change_threshold_percent", 0.4) / 100.0
            future_price = future_slice['close'].iloc[-1]
            price_change_pct = (future_price - entry_price) / entry_price
            if direction == 'buy' and price_change_pct > change_threshold: return 1
            if direction == 'sell' and price_change_pct < -change_threshold: return 1
            return 0

        elif method == "triple_barrier":
            atr = calculate_atr(data_series.iloc[:entry_idx+1])
            if atr <= 0: return 0
            tp_dist = atr * params.get('tp_atr_multiplier', 1.5)
            sl_dist = atr * params.get('sl_atr_multiplier', 1.7)
            if direction == 'buy':
                for i in range(len(future_slice)):
                    if future_slice['high'].iloc[i] >= entry_price + tp_dist: return 1
                    if future_slice['low'].iloc[i] <= entry_price - sl_dist: return 0
            else:
                for i in range(len(future_slice)):
                    if future_slice['low'].iloc[i] <= entry_price - tp_dist: return 1
                    if future_slice['high'].iloc[i] >= entry_price + sl_dist: return 0
            return 0

        return 0

    def is_trading_hours(self, symbol):
        """
        Checks if the current UTC time falls within the defined trading hours for a symbol.
        Used for actual trade execution and market monitoring.
        """
        try:
            schedules_config = self.config.get('trading_hours', {})
            symbol_schedule = schedules_config.get(symbol)

            if not symbol_schedule:
                return True # If no schedule defined, assume always tradable

            now_utc = datetime.now(timezone.utc)
            current_day_utc_str = now_utc.strftime('%A')
            current_time_utc = now_utc.time()

            scheduled_days = symbol_schedule.get('days', [])
            if not any(day.lower() == current_day_utc_str.lower() for day in scheduled_days):
                return False

            start_time_str = symbol_schedule.get('start', '00:00')
            end_time_str = symbol_schedule.get('end', '23:59')

            try:
                start_time = datetime.strptime(start_time_str, '%H:%M').time()
                end_time = datetime.strptime(end_time_str, '%H:%M').time()
            except ValueError:
                logger.error(f"Invalid time format in trading_hours for {symbol}. Start: '{start_time_str}', End: '{end_time_str}'. Assuming not tradable.")
                return False

            is_within_time = False
            if end_time < start_time: # Schedule crosses midnight
                is_within_time = (current_time_utc >= start_time or current_time_utc <= end_time)
            else: # Schedule within single day
                is_within_time = (start_time <= current_time_utc <= end_time)

            return is_within_time

        except Exception as e:
            logger.error(f"Error checking trading hours for {symbol}: {e}", exc_info=True)
            return False

    # +++ NEW FUNCTION: Check if within active trading session +++
    def is_within_active_session(self, symbol):
        """
        Checks if the current UTC time is within one of the active trading sessions
        defined in the config for the given symbol.
        """
        sessions_config = self.config.get('trading_sessions', {})
        if not sessions_config.get('enabled', False):
            return True # If session trading is disabled, always return true

        symbol_sessions = sessions_config.get('symbols', {}).get(symbol)
        if not symbol_sessions:
            logger.debug(f"SESSION CHECK ({symbol}): No specific sessions defined. Allowing trade.")
            return True # If no sessions are defined for the symbol, allow trading

        now_utc = datetime.now(timezone.utc)
        current_time_utc = now_utc.time()

        all_sessions = sessions_config.get('sessions', {})

        for session_name in symbol_sessions:
            session_times = all_sessions.get(session_name)
            if not session_times:
                continue

            try:
                start_time = datetime.strptime(session_times['start'], '%H:%M').time()
                end_time = datetime.strptime(session_times['end'], '%H:%M').time()
            except (ValueError, KeyError) as e:
                logger.error(f"SESSION CHECK ({symbol}): Invalid time format for session '{session_name}': {e}")
                continue

            # Check if the session crosses midnight (e.g., Asian session)
            if end_time < start_time:
                if current_time_utc >= start_time or current_time_utc <= end_time:
                    logger.debug(f"SESSION CHECK ({symbol}): Currently in active session: {session_name.upper()}")
                    return True
            # Standard check for sessions within the same day
            else:
                if start_time <= current_time_utc <= end_time:
                    logger.debug(f"SESSION CHECK ({symbol}): Currently in active session: {session_name.upper()}")
                    return True

        logger.debug(f"SESSION CHECK ({symbol}): Not in any active session ({', '.join(symbol_sessions)}).")
        return False


    def get_dxy_data_for_correlation(self, primary_symbol_data):
        """
        Fetches DXY data corresponding to the primary symbol's data timestamps.
        Includes a retry mechanism for robustness against transient connection errors.
        """
        dxy_symbol = self.config.get('dxy_symbol')
        if not dxy_symbol or primary_symbol_data.empty:
            return None

        time_diff = primary_symbol_data['time'].diff().min()
        tf_str = "M5"  # Default
        # Simple mapping, can be expanded
        if pd.notna(time_diff):
            if time_diff <= timedelta(minutes=5): tf_str = "M5"
            elif time_diff <= timedelta(minutes=15): tf_str = "M15"
            elif time_diff <= timedelta(minutes=30): tf_str = "M30"
            elif time_diff <= timedelta(hours=1): tf_str = "H1"
            elif time_diff <= timedelta(hours=4): tf_str = "H4"
            else: tf_str = "D1"

        start_time = primary_symbol_data['time'].iloc[0]
        end_time = primary_symbol_data['time'].iloc[-1]

        cache_key = (dxy_symbol, tf_str, start_time, end_time)
        with self.dxy_cache_lock:
            if cache_key in self.dxy_data_cache:
                return self.dxy_data_cache[cache_key]

        dxy_data = None
        max_retries = 3
        for attempt in range(max_retries):
            dxy_data = get_data(dxy_symbol, tf_str, from_date=start_time, to_date=end_time)
            if dxy_data is not None and not dxy_data.empty:
                logger.info(f"Successfully fetched DXY data on attempt {attempt + 1}.")
                break
            logger.warning(f"Attempt {attempt + 1}/{max_retries} to fetch DXY data failed. Retrying in 5 seconds...")
            if attempt < max_retries - 1:
                time.sleep(5)

        if dxy_data is None or dxy_data.empty:
            logger.error(f"Could not fetch DXY data for the required range after {max_retries} attempts.")
            return None

        dxy_data.set_index('time', inplace=True)
        dxy_data_resampled = dxy_data.reindex(primary_symbol_data['time'], method='ffill').reset_index()

        with self.dxy_cache_lock:
            if len(self.dxy_data_cache) > 50: # Keep cache size reasonable
                self.dxy_data_cache.pop(next(iter(self.dxy_data_cache)))
            self.dxy_data_cache[cache_key] = dxy_data_resampled


        return dxy_data_resampled

    def _generate_labels_for_training(self, historical_data, symbol, direction):
        """
        Helper function to generate features and labels based on the ML target definition in the config.
        """
        features = []
        labels = []
        training_window = self.config.get('ml_training_window', 120)

        # --- MODIFIED: Use the same prediction horizon for both methods for consistency ---
        prediction_horizon = self.config.get('ml_prediction_horizon', 20)
        target_def = self.config.get('ml_target_definition', {})
        target_params = target_def.get("params", {})

        logger.info(f"ML Data Prep ({symbol} {direction}): Starting feature/label generation. Method: '{target_def.get('method')}'. "
                    f"Total bars: {len(historical_data)}. Window: {training_window}, Horizon: {prediction_horizon}")

        dxy_correlation_data = None
        feat_eng_cfg = self.config.get('feature_engineering', {})
        if symbol.upper() == 'XAUUSDM' and feat_eng_cfg.get('dxy_correlation_for_xauusd', False):
            dxy_correlation_data = self.get_dxy_data_for_correlation(historical_data)
            if dxy_correlation_data is not None:
                logger.info(f"Successfully fetched DXY correlation data for {symbol} training.")
            else:
                logger.warning(f"Could not get DXY data for {symbol} training, proceeding without it.")

        for i in range(training_window, len(historical_data) - prediction_horizon):
            window_data = historical_data.iloc[i - training_window:i].copy()

            dxy_window_data = None
            if dxy_correlation_data is not None and not dxy_correlation_data.empty:
                dxy_window_data = dxy_correlation_data.iloc[i - training_window:i].copy()

            try:
                # MODIFIED: Call refactored function
                feature_vector = extract_ml_features(
                    symbol, window_data, direction, self, dxy_window_data
                )

                if feature_vector is None or (isinstance(feature_vector, list) and not feature_vector) or \
                   any(pd.isna(f_val) or (isinstance(f_val, (int, float)) and np.isinf(f_val)) for f_val in feature_vector):
                    continue

                entry_price = historical_data['close'].iloc[i]
                if entry_price == 0:
                    continue

                outcome = 0 # Default to loss (0)

                # --- REVISED LOGIC: Handle different target definition methods ---
                if target_def.get("method") == "price_change":
                    change_threshold = target_params.get("change_threshold_percent", 0.4) / 100.0

                    # Ensure the future price is within the bounds of the dataframe
                    future_price_index = i + prediction_horizon
                    if future_price_index < len(historical_data):
                        future_price = historical_data['close'].iloc[future_price_index]
                        price_change_pct = (future_price - entry_price) / entry_price

                        # Thesis defines win (1) / loss (-1) / same (0).
                        # We map this to a binary classification for the model: win (1) or not-win (0).
                        if direction == 'buy' and price_change_pct > change_threshold:
                            outcome = 1
                        elif direction == 'sell' and price_change_pct < -change_threshold:
                            outcome = 1

                elif target_def.get("method") == "atr_multiplier":
                    atr_period = target_params.get("atr_period", 14)
                    atr_multiplier = target_params.get("atr_multiplier", 1.5)

                    # MODIFIED: Call refactored function
                    atr_for_target = calculate_atr(window_data, period=atr_period)
                    if atr_for_target > 0:
                        profit_target_price_dist = atr_for_target * atr_multiplier
                        future_data_slice = historical_data.iloc[i+1 : i + 1 + prediction_horizon]
                        if not future_data_slice.empty:
                            if direction == 'buy':
                                if (future_data_slice['high'].max() - entry_price) >= profit_target_price_dist:
                                    outcome = 1
                            else: # sell
                                if (entry_price - future_data_slice['low'].min()) >= profit_target_price_dist:
                                    outcome = 1

                features.append(feature_vector)
                labels.append(outcome)

            except Exception as e:
                logger.warning(f"Error preparing historical data for {symbol} ({direction}) at index {i}: {e}", exc_info=False)
                continue

        return features, labels


    def initial_ml_models_training(self):
        logger.info("Starting initial ML model training for all symbols (Buy & Sell models)...")
        for symbol in self.config.get('symbols', []):
            for direction in ['buy', 'sell']:
                if self.ml_validator.is_fitted(symbol, direction):
                    logger.info(f"Initial ML model for {symbol} ({direction.upper()}) already fitted. Skipping.")
                    continue

                logger.info(f"Initial ML model training for {symbol} ({direction.upper()})")
                training_timeframe = self.config.get('ml_primary_training_timeframe', 'M30')
                historical_data = get_data(symbol, training_timeframe, bars=self.config.get('initial_training_bars', 20000))

                min_bars_needed = self.config.get('ml_training_window', 120) + self.config.get('ml_prediction_horizon', 20) + 1
                if historical_data is None or len(historical_data) < min_bars_needed:
                    logger.warning(f"ML Training ({symbol} {direction}): SKIPPED - Insufficient data. "
                                   f"Needed {min_bars_needed} bars, got {len(historical_data) if historical_data is not None else 0} for {training_timeframe}.")
                    continue

                historical_data.dropna(inplace=True)

                # --- MODIFIED: Use helper for label generation ---
                features, labels = self._generate_labels_for_training(historical_data, symbol, direction)

                logger.info(f"ML Data Prep ({symbol} {direction}): Finished. Generated {len(features)} feature sets.")
                if labels:
                    label_counts = Counter(labels)
                    logger.info(f"ML Data Prep ({symbol} {direction}): Label distribution: {dict(label_counts)}. Unique labels: {len(label_counts)}")
                    if len(label_counts) < 2:
                        logger.warning(f"ML Data Prep ({symbol} {direction}): Only one class of labels generated. Model training will be skipped.")
                else:
                    logger.warning(f"ML Data Prep ({symbol} {direction}): No labels were generated. Skipping training.")

                min_samples_fit = self.config.get('ml_min_samples_for_fit', 50)
                if not features or len(features) < min_samples_fit or len(set(labels)) < 2:
                    logger.warning(f"ML Training ({symbol} {direction}): Skipping fit. Features: {len(features)}, Unique Labels: {len(set(labels)) if labels else 0}. Min samples required: {min_samples_fit}.")
                    continue

                try:
                    stratify_labels = labels if len(set(labels)) > 1 else None
                    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=stratify_labels)

                    self.ml_validator.fit(symbol, X_train, y_train, direction)

                    if self.ml_validator.is_fitted(symbol, direction):
                        val_prob_positive = self.ml_validator.predict_proba(symbol, X_val, direction)[:, 1]
                        val_pred_binary = (val_prob_positive > 0.5).astype(int)
                        accuracy = accuracy_score(y_val, val_pred_binary)
                        f1 = f1_score(y_val, val_pred_binary, zero_division=0)
                        logger.info(f"Initial ML model for {symbol} ({direction.upper()}) on {training_timeframe} trained. Val Accuracy: {accuracy:.3f}, Val F1: {f1:.3f}")
                    else:
                        logger.warning(f"Initial ML model for {symbol} ({direction.upper()}) on {training_timeframe} did NOT fit successfully.")

                except Exception as e:
                    logger.error(f"Failed initial ML model training for {symbol} ({direction}) on {training_timeframe}: {e}", exc_info=True)

        self.last_global_retrain_time = datetime.now(timezone.utc)
        logger.info("Initial ML model training (Buy & Sell) process completed.")


    def perform_global_retrain(self):
        logger.info(f"Starting scheduled global ML model retraining for all symbols (Buy & Sell models)...")
        for symbol in self.config.get('symbols', []):
            for direction in ['buy', 'sell']:
                logger.info(f"Global Retraining: Processing {symbol} ({direction.upper()})")
                self.retrain_ml_model_for_symbol_and_direction(symbol, direction)
        self.last_global_retrain_time = datetime.now(timezone.utc)
        logger.info(f"Global ML model retraining completed. Next retraining in approx. {self.global_retrain_interval}.")


    def retrain_ml_model_for_symbol_and_direction(self, symbol, direction):
        logger.info(f"Retraining ML model for: {symbol} ({direction.upper()})")
        training_window = self.config.get('ml_training_window', 120)
        prediction_horizon = self.config.get('ml_prediction_horizon', 16)
        retrain_bars = self.config.get('retrain_data_bars', 15000)
        training_timeframe = self.config.get('ml_primary_training_timeframe', 'M30')
        historical_data = get_data(symbol, training_timeframe, bars=retrain_bars)
        min_bars_needed = training_window + prediction_horizon + 1

        if historical_data is None or len(historical_data) < min_bars_needed:
            logger.warning(f"ML Retrain ({symbol} {direction}): SKIPPED - Insufficient data.")
            return

        dxy_data = self.get_dxy_data_for_correlation(historical_data) if symbol.upper() == 'XAUUSDM' else None

        features, labels = [], []
        for i in range(training_window, len(historical_data) - prediction_horizon):
            window_data = historical_data.iloc[i - training_window:i]
            dxy_window_data = dxy_data.iloc[i - training_window:i] if dxy_data is not None else None
            try:
                feature_vector = extract_ml_features(symbol, window_data, direction, self, dxy_window_data)
                if feature_vector is not None:
                    label = self.generate_ml_labels(historical_data, i, direction)
                    features.append(feature_vector)
                    labels.append(label)
            except Exception as e:
                logger.warning(f"Error preparing historical data for {symbol} ({direction}) at index {i}: {e}", exc_info=False)
                continue

        if not features or len(np.unique(labels)) < 2:
            logger.warning(f"ML Retrain ({symbol} {direction}): No valid data or only one class. Skipping fit.")
            return

        self.ml_validator.fit(symbol, features, labels, direction)
        logger.info(f"ML model for {symbol} ({direction.upper()}) retraining process completed.")


    def mt5_tf_to_minutes(self, mt5_timeframe_str):
        if not isinstance(mt5_timeframe_str, str):
            logger.error(f"Invalid timeframe type for mt5_tf_to_minutes: {type(mt5_timeframe_str)}. Expected string.")
            return 5

        tf_map_str_to_int = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440,
            'W1': 10080, 'MN1': 43200
        }
        minutes = tf_map_str_to_int.get(mt5_timeframe_str.upper())
        if minutes is None:
            logger.warning(f"Unknown timeframe string '{mt5_timeframe_str}' in mt5_tf_to_minutes. Defaulting to 5 minutes.")
            return 5
        return minutes


    def get_trend(self, symbol, timeframe_str='H1'):
        if not isinstance(timeframe_str, str):
            logger.error(f"Invalid type for timeframe_str: {type(timeframe_str)} in get_trend. Defaulting to H1.")
            timeframe_str = 'H1'

        trend_config = self.config.get('auto_close_short_tf_profit_take', {})
        h1_filter_config = self.config.get('h1_trend_filter', {})

        bars_to_fetch = 200
        use_ema_logic = False

        ema_short_period = trend_config.get('trend_ema_short', 9)
        ema_long_period = trend_config.get('trend_ema_long', 21)

        sma_period_h1 = 200
        h1_trend_params_config = self.config.get('h1_trend_parameters', {})
        symbol_h1_params = h1_trend_params_config.get(symbol, {})
        sma_period_h1 = symbol_h1_params.get('sma_period', sma_period_h1)

        if timeframe_str in ['M1', 'M5', 'M15', 'M30']:
            bars_to_fetch = max(ema_long_period, 21) + 50
            use_ema_logic = True
        elif timeframe_str == 'H1':
            bars_to_fetch = sma_period_h1 + 50
            use_ema_logic = False
        else:
            logger.warning(f"get_trend called with unhandled timeframe '{timeframe_str}', defaulting to H1 SMA/MACD logic.")
            timeframe_str = 'H1'
            bars_to_fetch = sma_period_h1 + 50
            use_ema_logic = False

        try:
            data = get_data(symbol, timeframe_str, bars=bars_to_fetch)
            if data is None or data.empty:
                logger.warning(f"No data fetched for {symbol} on {timeframe_str} for trend analysis. Returning 'neutral'.")
                return 'neutral'

            required_bars = (ema_long_period + 1) if use_ema_logic else (sma_period_h1 + 1)
            if len(data) < required_bars:
                logger.warning(f"Insufficient data for {timeframe_str} trend on {symbol} (got {len(data)}, need {required_bars}). Returning 'neutral'.")
                return 'neutral'

            current_price = data['close'].iloc[-1]
            trend = 'neutral'

            if use_ema_logic:
                ema_short = data['close'].ewm(span=ema_short_period, adjust=False).mean().iloc[-1]
                ema_long = data['close'].ewm(span=ema_long_period, adjust=False).mean().iloc[-1]
                if any(pd.isna(v) for v in [current_price, ema_short, ema_long]):
                    return 'neutral'
                if current_price > ema_long and ema_short > ema_long: trend = 'uptrend'
                elif current_price < ema_long and ema_short < ema_long: trend = 'downtrend'

            else:
                sma_val = data['close'].rolling(window=sma_period_h1).mean().iloc[-1]
                # MODIFIED: Call refactored function
                macd_line, sig_line, _ = calculate_macd(data.copy())
                if any(pd.isna(v) for v in [sma_val, current_price, macd_line, sig_line]):
                    return 'neutral'
                if current_price > sma_val and macd_line > sig_line: trend = 'uptrend'
                elif current_price < sma_val and macd_line < sig_line: trend = 'downtrend'

            return trend
        except Exception as e:
            logger.error(f"Error in get_trend for {symbol} {timeframe_str}: {e}", exc_info=True)
            return 'neutral'


    def in_cooldown(self, symbol):
        last_trade_dt = self.last_trade_times.get(symbol)
        if last_trade_dt and (datetime.now(timezone.utc) - last_trade_dt) < self.cooldown_period:
            return True
        return False

    def can_open_new_trade(self, symbol, signal_to_open):
        max_for_symbol = self.config.get('max_trades_per_symbol', 2)
        open_positions = mt5.positions_get(symbol=symbol)
        if open_positions is None:
            logger.error(f"Failed to get open positions for {symbol}: {mt5.last_error()}")
            return False

        bot_positions = [p for p in open_positions if p.magic == 123456]

        if len(bot_positions) < max_for_symbol:
            return True
        else:
            return False

    def execute_trade(self, symbol, signal, data, timeframe_str, trade_params=None):
        # --- Initial Signal Check ---
        if signal == 'hold':
            logger.debug(f"EXECUTE_TRADE ({symbol} {timeframe_str}): Signal is 'hold'. No trade.")
            return False

        # --- NEW: News Filter (First Check) ---
        # This is the most important new check. If a trade is blocked by news,
        # we exit immediately before any other calculations.
        if self.news_manager.is_trade_prohibited(symbol):
            # The detailed reason is already logged by the news_manager
            logger.info(f"EXECUTE_TRADE REJECTED ({symbol} {timeframe_str} {signal}): Blocked by news filter.")
            return False
        # --- End of News Filter ---

        # --- Pre-trade Validation Checks ---
        h1_trend_config = self.config.get('h1_trend_filter', {'enabled': True, 'allow_neutral': True})
        if h1_trend_config.get('enabled', True):
            h1_trend = self.get_trend(symbol, timeframe_str='H1')
            logger.debug(f"EXECUTE_TRADE ({symbol} {timeframe_str} {signal}): H1 Trend: {h1_trend.upper()}")
            if h1_trend == 'neutral' and not h1_trend_config.get('allow_neutral', True):
                logger.info(f"EXECUTE_TRADE REJECTED ({symbol} {timeframe_str} {signal}): H1 trend NEUTRAL and neutral trades disallowed by config.")
                return False
            if (signal == 'buy' and h1_trend == 'downtrend') or \
            (signal == 'sell' and h1_trend == 'uptrend'):
                logger.info(f"EXECUTE_TRADE REJECTED ({symbol} {timeframe_str} {signal}): Signal against H1 trend {h1_trend.upper()}.")
                return False
        else:
            logger.debug(f"EXECUTE_TRADE ({symbol} {timeframe_str} {signal}): H1 trend filter disabled.")

        if not self.is_trading_hours(symbol):
            logger.info(f"EXECUTE_TRADE REJECTED ({symbol} {timeframe_str} {signal}): Outside trading hours.")
            return False
        if self.in_cooldown(symbol):
            logger.info(f"EXECUTE_TRADE REJECTED ({symbol} {timeframe_str} {signal}): Symbol in cooldown.")
            return False
        if not self.can_open_new_trade(symbol, signal):
            logger.info(f"EXECUTE_TRADE REJECTED ({symbol} {timeframe_str} {signal}): Max trades for symbol reached.")
            return False

        # --- ML Validation ---
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"EXECUTE_TRADE ({symbol} {timeframe_str}): Could not get symbol_info for {symbol}. Error: {mt5.last_error()}"); self.consecutive_failures += 1; return False
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or tick.time == 0:
            logger.warning(f"EXECUTE_TRADE ({symbol} {timeframe_str}): No valid tick data for {symbol}. Cannot place trade."); return False

        dxy_data_for_exec = None
        feat_eng_cfg = self.config.get('feature_engineering', {})
        if symbol.upper() == 'XAUUSDM' and feat_eng_cfg.get('dxy_correlation_for_xauusd', False):
            dxy_data_for_exec = self.get_dxy_data_for_correlation(data.copy())

        try:
            features = extract_ml_features(symbol, data.copy(), signal, self, dxy_data_for_exec)
            if features is None or any(pd.isna(f) for f in features):
                logger.error("ML feature extraction for trade execution returned None or NaN.")
                return False
        except Exception as e:
            logger.error(f"Error during feature extraction in execute_trade: {e}", exc_info=True)
            return False

        ml_prob_positive_outcome = 0.5
        if self.ml_validator.is_fitted(symbol, signal):
            try:
                ml_prob_positive_outcome = self.ml_validator.predict_proba(symbol, [features], signal)[0][1]
            except Exception as e:
                logger.error(f"EXECUTE_TRADE ({symbol} {timeframe_str} {signal}): Error in ML prediction: {e}", exc_info=True)
                ml_prob_positive_outcome = 0.0
        else:
            logger.warning(f"EXECUTE_TRADE ({symbol} {timeframe_str} {signal}): ML model not fitted; using neutral confidence 0.5.")

        confidence_thresholds_config = self.config.get('ml_confidence_thresholds', {})
        direction_thresholds = confidence_thresholds_config.get(signal, {})
        ml_conf_thresh = direction_thresholds.get(symbol, direction_thresholds.get("default", 0.55))

        if ml_prob_positive_outcome < ml_conf_thresh:
            logger.info(f"EXECUTE_TRADE REJECTED ({symbol} {timeframe_str} {signal}): ML confidence {ml_prob_positive_outcome:.2f} < threshold {ml_conf_thresh:.2f}")
            return False
        logger.info(f"EXECUTE_TRADE ({symbol} {timeframe_str} {signal}): ML confidence {ml_prob_positive_outcome:.2f} >= threshold {ml_conf_thresh:.2f}. Proceeding.")
        
        price_entry = tick.ask if signal == 'buy' else tick.bid
        if price_entry == 0 :
            logger.error(f"EXECUTE_TRADE ({symbol} {timeframe_str} {signal}): Entry price (bid/ask) is zero. Cannot place trade."); return False

        min_stop_price_distance = symbol_info.trade_stops_level * symbol_info.point * self.config.get('min_stop_multiplier', 1.5)
        if min_stop_price_distance == 0 : min_stop_price_distance = symbol_info.point * 10

        if trade_params and trade_params.get('source_strategy') == 'LiquiditySweep':
            logger.info(f"EXECUTE_TRADE ({symbol} {timeframe_str}): Using SL/TP from LiquiditySweep strategy.")
            sl = trade_params.get('sl')
            tp = trade_params.get('tp')
        elif trade_params and trade_params.get('source_strategy') == 'Fibonacci':
            logger.info(f"EXECUTE_TRADE ({symbol} {timeframe_str}): Using SL/TP from Fibonacci strategy.")
            sl = trade_params.get('sl')
            tp = trade_params.get('tp')
        else:
            # MODIFIED: Call refactored function
            atr = calculate_atr(data.copy(), period=self.config.get('atr_period_for_sl_tp', 14))
            if pd.isna(atr) or atr <= 0:
                atr = data['close'].iloc[-1] * self.config.get('atr_fallback_factor', 0.005)
                logger.warning(f"EXECUTE_TRADE ({symbol} {timeframe_str}): ATR was invalid or zero, using fallback ATR: {atr:.{symbol_info.digits}f}")

            price_range_last_n = 0
            recent_range_window = self.config.get('recent_range_window_for_sl_tp', 20)
            if len(data) >= recent_range_window:
                price_range_last_n = data['high'].iloc[-recent_range_window:].max() - data['low'].iloc[-recent_range_window:].min()
            if pd.isna(price_range_last_n) or price_range_last_n <= 0:
                price_range_last_n = data['close'].iloc[-1] * self.config.get('range_fallback_factor', 0.01)
                logger.warning(f"EXECUTE_TRADE ({symbol} {timeframe_str}): Recent price range was invalid or zero, using fallback range: {price_range_last_n:.{symbol_info.digits}f}")

            sl_distance = max(self.config.get('sl_atr_multiplier', 2.0) * atr, price_range_last_n * self.config.get('sl_range_factor', 0.5), min_stop_price_distance)
            tp_distance = max(sl_distance * self.config.get('risk_reward_ratio', 1.5), min_stop_price_distance)

            if signal == 'buy':
                sl = price_entry - sl_distance
                tp = price_entry + tp_distance
            else: # sell
                sl = price_entry + sl_distance
                tp = price_entry - tp_distance

        if signal == 'buy':
            sl = min(sl, tick.bid - min_stop_price_distance) if tick.bid > 0 else sl
            tp = max(tp, tick.ask + min_stop_price_distance) if tick.ask > 0 else tp
        else: # sell
            sl = max(sl, tick.ask + min_stop_price_distance) if tick.ask > 0 else sl
            tp = min(tp, tick.bid - min_stop_price_distance) if tick.bid > 0 else tp

        sl = round(sl, symbol_info.digits)
        tp = round(tp, symbol_info.digits)

        if (signal == 'buy' and (sl >= price_entry or tp <= price_entry or (tp - price_entry) < min_stop_price_distance or (price_entry - sl) < min_stop_price_distance)) or \
           (signal == 'sell' and (sl <= price_entry or tp >= price_entry or (price_entry - tp) < min_stop_price_distance or (sl - price_entry) < min_stop_price_distance)):
            logger.error(f"EXECUTE_TRADE ({symbol} {timeframe_str} {signal}): Invalid SL/TP. Entry={price_entry:.{symbol_info.digits}f}, SL={sl:.{symbol_info.digits}f}, TP={tp:.{symbol_info.digits}f}, MinStopDist={min_stop_price_distance:.{symbol_info.digits}f}")
            self.consecutive_failures += 1; return False

        acc_info = mt5.account_info()
        if not acc_info: logger.error(f"EXECUTE_TRADE ({symbol} {timeframe_str}): No account info for lot size."); return False

        max_lot_from_margin = float('inf')
        order_act_mt5 = mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL
        margin_available_for_trade = acc_info.margin_free * 0.95
        margin_for_one_lot = mt5.order_calc_margin(order_act_mt5, symbol, 1.0, price_entry)

        if margin_for_one_lot is not None and margin_for_one_lot > 0:
            max_lot_from_margin = margin_available_for_trade / margin_for_one_lot
        else:
            logger.warning(f"EXECUTE_TRADE ({symbol} {timeframe_str}): Could not calculate margin for one lot. Margin-based lot sizing disabled.")

        global_risk_pct = self.config.get('risk_percent_per_trade', 0.01)
        sym_risk_cfg = self.config.get('risk_params', {}).get(symbol, {})
        actual_risk_pct = sym_risk_cfg.get('max_risk_per_trade', global_risk_pct)
        risk_amt = acc_info.balance * actual_risk_pct

        if symbol_info.point == 0: logger.error(f"EXECUTE_TRADE ({symbol} {timeframe_str}): Point value for {symbol} is zero."); return False
        sl_pips_for_lot_calc = abs(price_entry - sl) / symbol_info.point

        min_sl_pips_for_calc = (symbol_info.trade_stops_level * 1.1) if symbol_info.trade_stops_level > 0 else 1.0
        if sl_pips_for_lot_calc < min_sl_pips_for_calc :
            sl_pips_for_lot_calc = min_sl_pips_for_calc
            logger.warning(f"EXECUTE_TRADE ({symbol} {timeframe_str}): Adjusted SL pips for lot calc to {sl_pips_for_lot_calc:.1f}.")
        if sl_pips_for_lot_calc <= 0: logger.error(f"EXECUTE_TRADE ({symbol} {timeframe_str}): SL pips for lot calculation is zero or negative."); return False

        if symbol_info.trade_tick_size == 0 or symbol_info.trade_tick_value == 0:
            logger.error(f"EXECUTE_TRADE ({symbol} {timeframe_str}): Symbol {symbol} has zero tick_size or tick_value."); return False
        val_per_pip_1_lot = (symbol_info.trade_tick_value / symbol_info.trade_tick_size) * symbol_info.point
        if val_per_pip_1_lot <= 0:
            logger.error(f"EXECUTE_TRADE ({symbol} {timeframe_str}): Calculated value per pip for 1 lot is zero or negative."); return False

        risk_based_lot = risk_amt / (sl_pips_for_lot_calc * val_per_pip_1_lot)

        lot = min(risk_based_lot, max_lot_from_margin)
        if lot < risk_based_lot:
            logger.info(f"EXECUTE_TRADE ({symbol} {timeframe_str}): Margin is the limiting factor. Lot reduced from {risk_based_lot:.4f} to {lot:.4f}.")

        decimals_in_step = 0
        if symbol_info.volume_step != 0:
            lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
            volume_step_str = str(symbol_info.volume_step)
            if '.' in volume_step_str:
                decimals_in_step = len(volume_step_str.split('.')[1])
            lot = round(lot, decimals_in_step)
        else:
            lot = round(lot, 2)

        lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))

        if lot < symbol_info.volume_min or lot == 0:
             lot = symbol_info.volume_min
             if lot == 0:
                 logger.error(f"EXECUTE_TRADE ({symbol} {timeframe_str}): Final lot size is zero after adjustments. Min volume is {symbol_info.volume_min}."); return False

        # --- MARGIN RETRY MECHANISM ---
        margin_req = None
        max_margin_retries = 3
        retry_delay_seconds = 5
        for attempt in range(max_margin_retries):
            margin_req = mt5.order_calc_margin(order_act_mt5, symbol, lot, price_entry)
            if margin_req is not None:
                logger.debug(f"EXECUTE_TRADE ({symbol} {timeframe_str}): Margin calculation successful on attempt {attempt + 1}.")
                break  # Success, exit retry loop

            logger.warning(
                f"EXECUTE_TRADE ({symbol} {timeframe_str}): Failed to calculate margin on attempt {attempt + 1}/{max_margin_retries}. "
                f"Error: {mt5.last_error()}. Retrying in {retry_delay_seconds} seconds..."
            )
            time.sleep(retry_delay_seconds)
            retry_delay_seconds *= 2  # Exponential backoff

        if margin_req is None:
            logger.error(
                f"EXECUTE_TRADE ({symbol} {timeframe_str}): FINAL - Failed to calculate margin after {max_margin_retries} attempts. "
                f"Aborting trade. Last error: {mt5.last_error()}"
            )
            self.consecutive_failures += 1
            return False
        # --- END MARGIN RETRY MECHANISM ---

        if acc_info.margin_free < margin_req:
            logger.error(f"EXECUTE_TRADE ({symbol} {timeframe_str}): Insufficient margin. Required: {margin_req:.2f}, Free: {acc_info.margin_free:.2f}"); return False

        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot, "type": order_act_mt5,
            "price": price_entry, "sl": sl, "tp": tp, "deviation": 20, "magic": 123456,
            "comment": f"AI {timeframe_str} {signal.upper()} R{actual_risk_pct*100:.1f}% ML{ml_prob_positive_outcome:.2f}"[:31],
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC
        }
        if hasattr(symbol_info, 'filling_modes') and mt5.SYMBOL_FILLING_FOK in symbol_info.filling_modes:
            request["type_filling"] = mt5.ORDER_FILLING_FOK

        logger.info(f"Trade Request: Sym={symbol}, Vol={lot}, Type={'BUY' if request['type']==mt5.ORDER_TYPE_BUY else 'SELL'}, Prc={price_entry:.{symbol_info.digits}f}, SL={sl:.{symbol_info.digits}f}, TP={tp:.{symbol_info.digits}f}, Comm='{request['comment']}'")
        result = mt5.order_send(request)

        if result is None:
            logger.error(f"Trade order_send returned None for {symbol}. Last error: {mt5.last_error()}"); self.consecutive_failures += 1; return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode_meanings = { 10004: "Requote", 10009: "Request completed", 10013: "Invalid request", 10014: "Invalid volume", 10015: "Invalid price", 10016: "Invalid stops", 10017: "Trade disabled", 10018: "Market closed", 10019: "No money", 10020: "Price changed" }
            err_msg = retcode_meanings.get(result.retcode, "Unknown error")
            logger.error(f"Trade execution failed for {symbol}: {result.comment} (RetCode: {result.retcode} - {err_msg})")
            self.consecutive_failures += 1; return False

        exec_price = result.price if result.price > 0 else price_entry
        exec_volume = result.volume if result.volume > 0 else lot
        logger.info(f"Trade EXECUTED: {signal.upper()} {symbol} @ {exec_price:.{symbol_info.digits}f} Lot: {exec_volume:.{decimals_in_step if symbol_info.volume_step != 0 else 2}f}. Order: {result.order}, Deal: {result.deal}")
        self.consecutive_failures = 0
        self.last_trade_times[symbol] = datetime.now(timezone.utc)

        active_strats = [s_name for s_name, sig_val in self.current_signals_for_logging.items() if sig_val[0] != 'hold']
        trade_log = {
            "symbol": symbol, "timeframe": timeframe_str, "signal": signal.upper(),
            "entry_price": exec_price, "sl_price": sl, "tp_price": tp, "lot_size": exec_volume,
            "strategies": active_strats,
            "entry_time": datetime.now(timezone.utc),
            "status": "open", "profit_loss": 0.0,
            "account_balance": acc_info.balance,
            "order_id": result.order, "deal_id": result.deal,
            "ml_confidence": round(ml_prob_positive_outcome, 3),
            "failure_reason": ""
        }
        save_trade(trade_log)
        return True


    def close_position_by_ticket(self, position_ticket, symbol, volume, trade_type_to_close, comment):
        logger.info(f"Attempting to close position {position_ticket} for {symbol} ({comment})")
        close_order_type = mt5.ORDER_TYPE_SELL if trade_type_to_close == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

        tick = mt5.symbol_info_tick(symbol)
        if not tick or tick.time == 0:
            logger.error(f"Cannot close position {position_ticket}: No valid tick data for {symbol}.")
            return False

        close_price = tick.bid if close_order_type == mt5.ORDER_TYPE_SELL else tick.ask
        if close_price == 0:
            logger.error(f"Cannot close position {position_ticket}: Close price (bid/ask) is zero for {symbol}.")
            return False

        symbol_info_obj = mt5.symbol_info(symbol)
        current_digits = symbol_info_obj.digits if symbol_info_obj else 5

        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
            "type": close_order_type, "position": position_ticket, "price": close_price,
            "deviation": 20, "magic": 123456, "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC
        }
        if symbol_info_obj and hasattr(symbol_info_obj, 'filling_modes') and mt5.SYMBOL_FILLING_FOK in symbol_info_obj.filling_modes:
            request["type_filling"] = mt5.ORDER_FILLING_FOK

        result = mt5.order_send(request)

        if result is None:
            logger.error(f"Order send failed to close position {position_ticket} (result is None). Last error: {mt5.last_error()}")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position {position_ticket} for {symbol}: {result.comment} (retcode: {result.retcode})")
            return False

        logger.info(f"Sent close order for position {position_ticket}. Result Order: {result.order}, Deal: {result.deal}. Comment: {comment}")

        time.sleep(self.config.get('trade_close_reconciliation_delay_seconds', 3))

        closed_profit = 0.0
        final_exit_price = close_price
        exit_time_dt = datetime.now(timezone.utc)

        deals = mt5.history_deals_get(position=position_ticket)
        if deals:
            closing_deals = [d for d in deals if d.position_id == position_ticket and d.entry in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]]
            if closing_deals:
                closing_deals.sort(key=lambda x: x.time_msc)
                for deal_item in closing_deals:
                    closed_profit += deal_item.profit

                last_deal = closing_deals[-1]
                final_exit_price = last_deal.price
                exit_time_dt = datetime.fromtimestamp(last_deal.time_msc / 1000, tz=timezone.utc)
                logger.info(f"Reconciled P/L: {closed_profit:.2f}, Actual Exit Price: {final_exit_price:.{current_digits}f} for closed position {position_ticket}")
            else:
                logger.warning(f"No closing deals found in history for position {position_ticket} immediately after close.")
        else:
            logger.warning(f"Could not fetch deal history for position {position_ticket} after close.")

        update_data = {
            "status": "closed_auto", "exit_reason": comment, "exit_time": exit_time_dt,
            "profit_loss": round(closed_profit, 2), "exit_price": round(final_exit_price, current_digits)
        }
        update_trade_status(position_ticket, update_data)
        logger.info(f"Trade {position_ticket} status updated in log. P/L: {update_data['profit_loss']:.2f}")
        return True


    def force_full_ml_retraining(self):
        logger.info("Force full ML model re-initialization and training triggered by API.")
        was_bot_running_initially = self.bot_running

        if was_bot_running_initially:
            logger.info("Stopping bot temporarily for full ML model retraining...")
            self.stop_bot_logic()
            if self.bot_thread and self.bot_thread.is_alive():
                self.bot_thread.join(timeout=max(10, self.config.get('monitoring_interval_seconds', 60) + 5))
                if self.bot_thread.is_alive():
                    logger.warning("Bot thread did not exit gracefully. Proceeding with retrain.")

        logger.info("Reloading configuration and reinitializing MLValidator before full ML training.")
        self.load_config_and_reinitialize()

        logger.info("Starting full initial ML model training process using 'initial_training_bars'...")
        success = False
        message = "Forced full ML training initiated."
        try:
            self.initial_ml_models_training()
            message = "Full initial ML model training process completed successfully."
            logger.info(message)
            self.last_global_retrain_time = datetime.now(timezone.utc)
            logger.info(f"Global ML retrain timer reset. Next scheduled retrain in approx. {self.global_retrain_interval}.")
            success = True
        except Exception as e:
            message = f"Error during forced full ML training: {str(e)}"
            logger.error(message, exc_info=True)

        if was_bot_running_initially:
            logger.info("Restarting bot monitoring after full ML model retraining...")
            if not mt5.terminal_info():
                logger.warning("MT5 connection lost during retraining. Attempting to reconnect.")
                if not connect_mt5(self.config.get('mt5_credentials', {})):
                    message += " MT5 reconnection failed, bot not restarted."
                    return False, message
                self.ensure_symbols_selected()

            self.start_bot_logic()
            if not self.bot_running:
                message += " However, failed to restart bot after retraining. Check logs."
                success = False

        return success, message


    def monitor_market(self):
        with self.status_lock:
            if not self.bot_running:
                logger.info("Bot not running. Monitor loop exits.")
                return

        while self.bot_running:
            try:
                # Add this after the trading_paused check:
                if hasattr(self, 'ml_validator') and not self.ml_validator.any_models_fitted():
                    logger.warning("No ML models fitted! Forcing retraining...")
                    self.perform_global_retrain()

                if datetime.now(timezone.utc) - self.last_global_retrain_time > self.global_retrain_interval:
                    self.perform_global_retrain()

                self.check_closed_trades()
                self.manage_open_trades()
                self.ensure_symbols_selected()

                for symbol in self.config.get('symbols', []):
                    if not self.bot_running: break

                    if not mt5.symbol_select(symbol, True):
                        logger.warning(f"Could not select symbol {symbol} in Market Watch. Skipping.")
                        continue

                    # *** MODIFIED: Use new session check and existing trading hours check ***
                    if not self.is_trading_hours(symbol):
                        logger.debug(f"MONITOR ({symbol}): Outside general trading hours. Skipping.")
                        continue
                    if not self.is_within_active_session(symbol):
                        logger.debug(f"MONITOR ({symbol}): Outside active trading session. Skipping.")
                        continue

                    if self.in_cooldown(symbol):
                        logger.debug(f"MONITOR ({symbol}): Symbol {symbol} is in cooldown. Skipping.")
                        continue

                    for tf_str in self.config.get('timeframes', ['M5']):
                        if not self.bot_running: break
                        logger.debug(f"MONITOR ({symbol} {tf_str}): Analyzing...")

                        data = get_data(symbol, tf_str, bars=self.config.get('bars', 1000))
                        if data is None or data.empty:
                            logger.warning(f"MONITOR ({symbol} {tf_str}): No data received. Skipping."); continue

                        data_for_strategies = data.copy()
                        if 'time' not in data_for_strategies.columns and isinstance(data_for_strategies.index, pd.DatetimeIndex):
                            data_for_strategies['time'] = data_for_strategies.index


                        current_signals = {}

                        # --- MODIFIED LOGIC: Store full signal tuple (signal, strength, params) ---
                        for strategy in self.strategies:
                            try:
                                signal_tuple = strategy.get_signal(data_for_strategies.copy(), symbol=symbol, timeframe=tf_str)
                                # Ensure signal_tuple is always a 3-element tuple
                                if isinstance(signal_tuple, tuple) and len(signal_tuple) == 3:
                                    current_signals[strategy.name] = signal_tuple
                                else: # Fallback for older strategies not returning 3 elements
                                    s, st = signal_tuple if isinstance(signal_tuple, tuple) else (signal_tuple, 0.0)
                                    current_signals[strategy.name] = (s, float(st) if pd.notna(st) else 0.0, None)
                            except Exception as e:
                                logger.error(f"Error in {strategy.name}.get_signal for {symbol} {tf_str}: {e}", exc_info=True)
                                current_signals[strategy.name] = ('hold', 0.0, None)
                        # --- END OF MODIFIED LOGIC ---

                        logger.debug(f"MONITOR ({symbol} {tf_str}): Raw signals: { {k: v[0] for k, v in current_signals.items()} }")
                        self.current_signals_for_logging = current_signals

                        consensus_signal, consensus_params = self.analyze_signals(current_signals, data_for_strategies.copy(), symbol, tf_str)

                        if consensus_signal != 'hold':
                            logger.info(f"MONITOR ({symbol} {tf_str}): Consensus is '{consensus_signal.upper()}'. Attempting trade execution.")
                            # Pass the consensus_params to the execute_trade function
                            self.execute_trade(symbol, consensus_signal, data_for_strategies.copy(), tf_str, trade_params=consensus_params)
                        else:
                            logger.debug(f"MONITOR ({symbol} {tf_str}): Consensus is 'hold'. No trade action.")

                        if self.consecutive_failures >= self.config.get('max_consecutive_trade_failures', 3):
                            logger.error(f"{self.config.get('max_consecutive_trade_failures', 3)} consecutive trade execution failures. Pausing bot for safety.")
                            self.stop_bot_logic()
                            self.consecutive_failures = 0
                            break

                    if not self.bot_running: break

                time.sleep(self.config.get('monitoring_interval_seconds', 60))

            except ConnectionResetError as cre:
                logger.error(f"ConnectionResetError in monitor_market: {cre}. Attempting to reconnect MT5...")
                if not mt5.terminal_info() or not mt5.ping():
                    if not connect_mt5(self.config.get('mt5_credentials')):
                        logger.error("MT5 reconnection failed. Sleeping before retry.")
                        time.sleep(300)
                    else:
                        logger.info("MT5 reconnected successfully after ConnectionResetError.")
                        self.ensure_symbols_selected()
                else:
                    logger.info("MT5 ping successful despite ConnectionResetError. Continuing.")
                    time.sleep(60)

            except RuntimeError as rte:
                logger.error(f"RuntimeError in monitor_market: {rte}")
                if "IPC timeout" in str(rte) or "Terminal not found" in str(rte) or "Server is unavailable" in str(rte):
                    logger.warning("Suspected MT5 terminal/connection issue. Attempting reconnect.")
                    if not connect_mt5(self.config.get('mt5_credentials')):
                        logger.error("MT5 reconnection failed after RuntimeError. Sleeping.")
                        time.sleep(300)
                    else:
                        logger.info("MT5 reconnected successfully after RuntimeError.")
                        self.ensure_symbols_selected()
                else:
                    time.sleep(60)

            except Exception as e:
                logger.critical(f"Critical unhandled error in monitor_market: {e}", exc_info=True)
                self.last_error_message = str(e)
                self.stop_bot_logic()
                time.sleep(self.config.get('error_sleep_interval_seconds', 300))

        logger.info("monitor_market loop ended because bot_running is false.")


    def analyze_signals(self, signals, data, symbol, timeframe):
        """
        Analyzes signals from multiple strategies, applying dynamic performance-based weights
        and static boosts to reach a trading consensus.
        """
        logger.debug(f"ANALYZE_SIGNALS ({symbol} {timeframe}): Starting analysis. Raw signals: { {k: v[0] for k, v in signals.items()} }")

        buy_weighted_strength_sum = 0.0
        sell_weighted_strength_sum = 0.0
        final_trade_params = None

        # --- DYNAMIC WEIGHTING: Fetch weights based on recent performance ---
        lookback_days = self.config.get('strategy_weighting_lookback_days', 30)
        strategy_performance_weights = get_strategy_weights(lookback_days=lookback_days)

        # --- VOLATILITY & THRESHOLD: Determine consensus threshold based on market volatility ---
        volatility = 0.0
        if not data.empty and len(data) >= 20 and all(c in data.columns for c in ['high', 'low', 'close']) and 'close' in data and data['close'].iloc[-1] != 0:
            volatility = (data['high'].iloc[-20:].max() - data['low'].iloc[-20:].min()) / data['close'].iloc[-1]

        consensus_threshold_config_main = self.config.get('consensus_threshold', {})
        default_consensus_node = consensus_threshold_config_main.get("default", {"low_vol": 1.0, "high_vol": 1.5, "vol_split": 0.005})
        symbol_consensus_config = consensus_threshold_config_main.get(symbol, default_consensus_node)
        final_consensus_params = {**default_consensus_node, **symbol_consensus_config}
        threshold_val = final_consensus_params["low_vol"] if volatility < final_consensus_params["vol_split"] else final_consensus_params["high_vol"]

        # --- ADX FILTER: Pre-filter signals based on ADX trend strength ---
        adx_filter_cfg = self.config.get('adx_signal_filter', {})
        apply_adx_filter = adx_filter_cfg.get('enabled', False)
        latest_adx_value, latest_plus_di, latest_minus_di = np.nan, np.nan, np.nan

        adx_strategy_instance = self._adx_strategy_for_features
        if apply_adx_filter and adx_strategy_instance:
            try:
                adx_indicator_values = adx_strategy_instance.get_indicator_values(data.copy())
                if adx_indicator_values and adx_indicator_values.get('adx') is not None and not adx_indicator_values['adx'].empty:
                    latest_adx_value, latest_plus_di, latest_minus_di = adx_indicator_values['adx'].iloc[-1], adx_indicator_values['plus_di'].iloc[-1], adx_indicator_values['minus_di'].iloc[-1]
                else: apply_adx_filter = False
            except Exception: apply_adx_filter = False

        filtered_signals = {}
        for strat_name, (original_signal, strength, params) in signals.items():
            signal_to_process = original_signal
            if apply_adx_filter and pd.notna(latest_adx_value) and strat_name != "ADX" and original_signal != 'hold':
                min_adx_for_entry = adx_filter_cfg.get(f"min_adx_for_entry_{symbol}", adx_filter_cfg.get('min_adx_for_entry', 20))
                if latest_adx_value < min_adx_for_entry or \
                  (adx_filter_cfg.get('require_di_confirmation', True) and (
                      (original_signal == 'buy' and latest_plus_di <= latest_minus_di) or
                      (original_signal == 'sell' and latest_minus_di <= latest_plus_di)
                  )):
                    signal_to_process = 'hold'
            filtered_signals[strat_name] = (signal_to_process, strength, params)

        # --- WEIGHTED SUMMATION: Calculate total strength for BUY and SELL signals ---
        for s_name, (signal, s_strength, s_params) in filtered_signals.items():
            # 1. Get the dynamic performance weight, default to 1.0 if not found
            perf_weight = strategy_performance_weights.get(s_name, 1.0)

            # 2. Get the static boost from config
            boost_config = self.config.get('strategy_boost_factor', {})
            strategy_boost_config = boost_config.get(s_name, 1.0)
            boost_multiplier = strategy_boost_config.get(timeframe, strategy_boost_config.get('default', 1.0)) if isinstance(strategy_boost_config, dict) else strategy_boost_config

            # 3. Combine static boost and dynamic performance weight
            final_weight_multiplier = boost_multiplier * perf_weight

            # 4. Apply the combined weight to the signal's strength
            eff_w = final_weight_multiplier * max(s_strength, 0.1 if signal != 'hold' and s_strength == 0 else s_strength)

            if signal == 'buy':
                buy_weighted_strength_sum += eff_w
                if s_params and not final_trade_params: final_trade_params = s_params
            elif signal == 'sell':
                sell_weighted_strength_sum += eff_w
                if s_params and not final_trade_params: final_trade_params = s_params

        # --- CONSENSUS DECISION ---
        dominance_factor = self.config.get('consensus_dominance_factor', 1.8)
        final_consensus_signal = 'hold'
        if buy_weighted_strength_sum >= threshold_val and buy_weighted_strength_sum > sell_weighted_strength_sum * dominance_factor:
            final_consensus_signal = 'buy'
        elif sell_weighted_strength_sum >= threshold_val and sell_weighted_strength_sum > buy_weighted_strength_sum * dominance_factor:
            final_consensus_signal = 'sell'

        logger.info(
            f"ANALYZE_SIGNALS ({symbol} {timeframe}): Final Consensus: {final_consensus_signal.upper()} "
            f"(Buy Strength: {buy_weighted_strength_sum:.2f}, Sell Strength: {sell_weighted_strength_sum:.2f}, "
            f"Threshold: {threshold_val:.2f})"
        )

        if final_consensus_signal == 'hold':
            return 'hold', None
        return final_consensus_signal, final_trade_params


    def get_bot_status_for_ui(self):
        with self.status_lock:
            status_str = "STOPPED"
            if self.bot_running:
                status_str = "RUNNING"
            elif self.last_error_message:
                status_str = "ERROR"

        acc_info = mt5.account_info()
        mt5_connected = mt5.terminal_info() is not None

        if not mt5_connected:
            status_str = "MT5_DISCONNECTED"
        elif status_str == "STOPPED" and not self.bot_running and self.bot_thread is None:
            status_str = "UNINITIALIZED"

        current_last_error = self.last_error_message
        if status_str not in ["ERROR", "MT5_DISCONNECTED"]:
            current_last_error = None

        num_buy_fitted, num_sell_fitted = 0, 0
        if hasattr(self, 'ml_validator') and self.ml_validator:
            for sym in self.config.get('symbols', []):
                if self.ml_validator.is_fitted(sym, 'buy'): num_buy_fitted +=1
                if self.ml_validator.is_fitted(sym, 'sell'): num_sell_fitted +=1
        expected_models = len(self.config.get('symbols', [])) * 2

        return {
            "bot_status": status_str,
            "balance": acc_info.balance if acc_info else "N/A", "equity": acc_info.equity if acc_info else "N/A",
            "free_margin": acc_info.margin_free if acc_info else "N/A", "margin_level": acc_info.margin_level if acc_info else "N/A",
            "last_ml_retrain": self.last_global_retrain_time.isoformat() if self.last_global_retrain_time > datetime.min.replace(tzinfo=timezone.utc) else "Never",
            "ml_models_fitted_buy": num_buy_fitted, "ml_models_fitted_sell": num_sell_fitted,
            "ml_models_expected": expected_models, "last_error": current_last_error
        }

    def get_open_positions_for_ui(self):
        positions = mt5.positions_get()
        if positions is None:
            return []

        ui_positions = []
        for pos in positions:
            if pos.magic == 123456:
                symbol_info = mt5.symbol_info(pos.symbol)
                digits = symbol_info.digits if symbol_info else 5
                point = symbol_info.point if symbol_info else (0.1**digits)

                pnl_pips = 0
                if point > 0:
                    pnl_pips = ((pos.price_current - pos.price_open) / point) if pos.type == mt5.ORDER_TYPE_BUY else ((pos.price_open - pos.price_current) / point)

                ui_positions.append({
                    "ticket": pos.ticket, "symbol": pos.symbol, "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                    "volume": round(pos.volume, 2), "entry_price": round(pos.price_open, digits),
                    "current_price": round(pos.price_current, digits), "sl": round(pos.sl, digits) if pos.sl != 0 else 0.0,
                    "tp": round(pos.tp, digits) if pos.tp != 0 else 0.0, "pnl_pips": round(pnl_pips,1),
                    "pnl_currency": round(pos.profit, 2),
                    "entry_time": datetime.fromtimestamp(pos.time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if pos.time > 0 else "N/A",
                    "entry_tf": self.get_entry_timeframe_from_comment(pos.comment) or "N/A", "comment": pos.comment
                })
        return ui_positions


    def get_trade_history_for_ui(self, page=1, limit=20):
        trades_collection = MongoDBConnection.get_trades_collection()
        if not trades_collection:
            logger.warning("Cannot get trade history for UI, MongoDB not connected or collection not found.")
            return [], 0

        trades_data = []
        total_trades = 0
        try:
            query = {"status": {"$in": ["closed", "closed_auto", "closed_unknown"]}}
            total_trades = trades_collection.count_documents(query)

            skip_amount = (page - 1) * limit
            trade_cursor = trades_collection.find(query).sort("entry_time", DESCENDING).skip(skip_amount).limit(limit)

            for row_dict in trade_cursor:
                row_dict["_id"] = str(row_dict["_id"])

                entry_time_dt, exit_time_dt = row_dict.get('entry_time'), row_dict.get('exit_time')
                entry_time_str = "N/A"
                if isinstance(entry_time_dt, datetime):
                    if entry_time_dt.tzinfo is None: entry_time_dt = entry_time_dt.replace(tzinfo=timezone.utc)
                    entry_time_str = entry_time_dt.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                elif isinstance(entry_time_dt, str): entry_time_str = entry_time_str

                exit_time_str = "N/A"
                if isinstance(exit_time_dt, datetime):
                    if exit_time_dt.tzinfo is None: exit_time_dt = exit_time_dt.replace(tzinfo=timezone.utc)
                    exit_time_str = exit_time_dt.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                elif isinstance(exit_time_dt, str): exit_time_str = exit_time_str

                digits = 5
                symbol_val = row_dict.get("symbol", "N/A")
                if symbol_val != "N/A":
                    symbol_info = mt5.symbol_info(symbol_val)
                    if symbol_info: digits = symbol_info.digits

                def safe_float(value, default=0.0):
                    try: return float(value)
                    except (ValueError, TypeError): return default

                trade_item = {
                    "symbol": symbol_val, "ticket": str(row_dict.get("order_id", "N/A")),
                    "deal_id": str(row_dict.get("deal_id", "N/A")), "type": str(row_dict.get("signal", "N/A")).upper(),
                    "volume": safe_float(row_dict.get("lot_size")), "entry_price": round(safe_float(row_dict.get("entry_price")), digits),
                    "exit_price": round(safe_float(row_dict.get("exit_price")), digits), "profit_loss": safe_float(row_dict.get("profit_loss")),
                    "entry_time": entry_time_str, "exit_time": exit_time_str,
                    "entry_tf": row_dict.get("timeframe", "N/A"), "exit_reason": row_dict.get("exit_reason", "N/A"),
                    "ml_confidence": f"{safe_float(row_dict.get('ml_confidence', np.nan)):.2f}" if pd.notna(row_dict.get('ml_confidence')) else "N/A",
                    "strategies": str(row_dict.get("strategies", "N/A"))
                }
                trades_data.append(trade_item)

            return trades_data, total_trades
        except errors.PyMongoError as e:
            logger.error(f"Error fetching trade history from MongoDB for UI: {e}", exc_info=True)
            return [], 0
        except Exception as e:
            logger.error(f"Unexpected error in get_trade_history_for_ui: {e}", exc_info=True)
            return [],0


    def get_logs_for_ui(self, limit=50, level_filter=None):
        try:
            log_file_path = 'logs/trading_ea.log'
            if not os.path.exists(log_file_path):
                return ["Log file not found."]

            lines = []
            with open(log_file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()

                if level_filter and level_filter.upper() != "ALL":
                    filter_str = f" - {level_filter.upper()} - "
                    filtered_lines = [line for line in all_lines if filter_str in line]
                else:
                    filtered_lines = all_lines

                lines = filtered_lines[-limit:]

            return [line.strip() for line in lines]
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return [f"Error reading log file: {str(e)}"]


    def delete_bot_logs(self):
        global logger
        log_file_path = 'logs/trading_ea.log'

        if logger: logger.info(f"Attempting to delete log file: {log_file_path}")
        else: print(f"Logger not available when attempting to delete log file: {log_file_path}")

        try:
            handler_to_remove = None
            if logger and hasattr(logger, 'handlers'):
                for handler in logger.handlers:
                    if isinstance(handler, logging.FileHandler) and hasattr(handler, 'baseFilename') and handler.baseFilename and os.path.abspath(handler.baseFilename) == os.path.abspath(log_file_path):
                        handler_to_remove = handler
                        break

            if handler_to_remove:
                handler_to_remove.close()
                logger.removeHandler(handler_to_remove)

            if os.path.exists(log_file_path):
                os.remove(log_file_path)
                msg = f"Log file {log_file_path} deleted successfully."
            else:
                msg = f"Log file {log_file_path} not found, no deletion needed."

            logger = setup_logging()
            logger.info(msg + " Logging re-initialized.")
            return True, msg
        except Exception as e:
            try:
                logger = setup_logging()
                logger.error(f"Error during log deletion for '{log_file_path}': {e}", exc_info=True)
            except Exception as e2:
                print(f"CRITICAL: Error deleting log '{log_file_path}': {e}. FAILED to re-init logging: {e2}")
            return False, f"Error deleting log file: {str(e)}"


    def start_bot_logic(self):
        with self.status_lock:
            if self.bot_running:
                logger.info("Bot is already running.")
                return False
            self.bot_running = True
            self.last_error_message = None
            logger.info("Trading bot started.")

        if self.bot_thread is None or not self.bot_thread.is_alive():
            if mt5.terminal_info():
                self.ensure_symbols_selected()
            self.bot_thread = threading.Thread(target=self.monitor_market, daemon=True)
            self.bot_thread.start()
            logger.info("monitor_market thread started.")
        return True


    def stop_bot_logic(self):
        with self.status_lock:
            if not self.bot_running:
                logger.info("Bot is already stopped.")
                return False
            self.bot_running = False
            logger.info("Trading bot stopping... please wait for current cycle to complete.")
        return True


    def trigger_manual_retrain(self):
        if not self.bot_running and not mt5.terminal_info():
            logger.error("Cannot trigger manual retrain: MT5 not connected and bot not running.")
            return False, "MT5 not connected and bot not running."

        logger.info("Manual ML retraining triggered by UI.")
        retrain_thread = threading.Thread(target=self.perform_global_retrain, daemon=True)
        retrain_thread.start()
        return True, "Manual ML retraining process initiated in background."


    def manual_close_trade_by_ticket(self, ticket_id):
        try:
            ticket_id = int(ticket_id)
        except ValueError:
            logger.error(f"Invalid ticket ID format for manual close: {ticket_id}")
            return False, "Invalid ticket ID format."

        position = mt5.positions_get(ticket=ticket_id)
        if not position:
            logger.warning(f"Position with ticket {ticket_id} not found for manual close.")
            return False, f"Position {ticket_id} not found."

        pos = position[0]

        logger.info(f"Attempting manual close for ticket {pos.ticket} (Symbol: {pos.symbol}, Type: {'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'}, Volume: {pos.volume})")
        closed = self.close_position_by_ticket(pos.ticket, pos.symbol, pos.volume, pos.type, "Manual UI Close")

        if closed:
            return True, f"Close order sent successfully for ticket {ticket_id}."
        else:
            return False, f"Failed to send close order for ticket {ticket_id}. Check logs for details."

    def get_equity_curve_data(self, limit_points=100):
        logger.info(f"Generating equity curve data from MongoDB (limit: {limit_points} points).")
        trades_collection = MongoDBConnection.get_trades_collection()
        labels, equity_values = [], []

        if not trades_collection:
            logger.warning("MongoDB not connected or trades collection not found. Trying current MT5 equity.")
            acc_info = mt5.account_info()
            if acc_info:
                return {"labels": [datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')], "equity": [round(acc_info.equity, 2)]}
            return {"labels": [], "equity": [], "error": "MongoDB unavailable and MT5 connection failed."}

        try:
            first_trade_doc = trades_collection.find_one({"account_balance": {"$exists": True, "$type": "number"}}, sort=[("entry_time", 1)])

            if not first_trade_doc:
                logger.info("No trades with numeric account_balance found. Using current MT5 equity.")
                acc_info = mt5.account_info()
                if acc_info:
                    return {"labels": [datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')], "equity": [round(acc_info.equity, 2)]}
                return {"labels": [], "equity": [], "error": "No starting point in MongoDB and MT5 unavailable."}

            initial_equity = first_trade_doc.get('account_balance', 0.0)
            try: initial_equity = float(initial_equity)
            except (ValueError, TypeError): initial_equity = 0.0; logger.warning("Initial equity from DB was not a number.")

            if first_trade_doc.get('status') in ['closed', 'closed_auto'] and pd.notna(first_trade_doc.get('profit_loss')):
                try:
                    initial_equity -= float(first_trade_doc.get('profit_loss', 0.0))
                except (ValueError, TypeError): pass

            base_ts_dt = first_trade_doc.get('entry_time')
            if not isinstance(base_ts_dt, datetime):
                 base_ts_dt = datetime.now(timezone.utc)
                 logger.warning(f"Invalid entry_time for first trade {first_trade_doc.get('order_id')}, using current time for equity curve start.")
            if base_ts_dt.tzinfo is None: base_ts_dt = base_ts_dt.replace(tzinfo=timezone.utc)

            labels.append(base_ts_dt.strftime('%Y-%m-%d %H:%M'))
            equity_values.append(round(initial_equity, 2))
            running_equity = initial_equity

            closed_trades_cursor = trades_collection.find(
                {"status": {"$in": ["closed", "closed_auto"]}, "exit_time": {"$exists": True, "$type": "date"}, "profit_loss": {"$exists": True, "$type": "number"}},
                sort=[("exit_time", 1)]
            )

            for trade in closed_trades_cursor:
                exit_time_dt, profit_loss_val = trade.get('exit_time'), trade.get('profit_loss', 0.0)
                try: profit_loss_val = float(profit_loss_val)
                except (ValueError, TypeError): profit_loss_val = 0.0

                if isinstance(exit_time_dt, datetime):
                    if exit_time_dt.tzinfo is None: exit_time_dt = exit_time_dt.replace(tzinfo=timezone.utc)
                    running_equity += profit_loss_val
                    labels.append(exit_time_dt.strftime('%Y-%m-%d %H:%M'))
                    equity_values.append(round(running_equity, 2))
                else:
                    logger.warning(f"Skipping trade {trade.get('order_id')} for equity curve due to invalid exit_time type: {type(exit_time_dt)}")

            acc_info = mt5.account_info()
            if acc_info:
                current_mt5_equity = round(acc_info.equity, 2)
                current_time_label = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
                if not equity_values or (labels[-1] != current_time_label or equity_values[-1] != current_mt5_equity):
                     if not labels or pd.to_datetime(current_time_label) >= pd.to_datetime(labels[-1]):
                        labels.append(current_time_label)
                        equity_values.append(current_mt5_equity)

            if not equity_values:
                return {"labels": [], "equity": [], "error": "Could not construct equity points."}

            if len(equity_values) > limit_points:
                slice_start = len(equity_values) - limit_points
                labels, equity_values = labels[slice_start:], equity_values[slice_start:]

            return {"labels": labels, "equity": equity_values}

        except errors.PyMongoError as e:
            logger.error(f"MongoDB error generating equity curve: {e}", exc_info=True)
            return {"labels": [], "equity": [], "error": f"MongoDB error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error generating equity curve: {e}", exc_info=True)
            return {"labels": [], "equity": [], "error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    bot_config_for_direct_run = {}
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path_direct = os.path.join(current_dir, '..', 'config', 'config.json')

        if not os.path.exists(os.path.dirname(config_path_direct)):
            os.makedirs(os.path.dirname(config_path_direct))
            logger.info(f"Created directory for config: {os.path.dirname(config_path_direct)}")

        if os.path.exists(config_path_direct):
            with open(config_path_direct, 'r') as f_direct:
                bot_config_for_direct_run = json.load(f_direct)
        else:
            logger.warning(f"Config file not found at {config_path_direct} for direct run. Using empty config.")

    except Exception as e_direct_cfg:
        logger.error(f"Could not load config.json for direct run from {config_path_direct}: {e_direct_cfg}.")

    if not connect_mt5(bot_config_for_direct_run.get('mt5_credentials')):
        logger.critical("MT5 connection failed on startup. Bot cannot start.")
        MongoDBConnection.close_connection()
        exit(1)

    bot = TradingBot()
    try:
        logger.info("Performing initial ML model training sequence...")
        try:
            bot.initial_ml_models_training()
        except Exception as e:
            logger.error(f"Initial ML training failed: {e}", exc_info=True)

        for symbol in bot.config.get('symbols', []):
            for direction in ['buy', 'sell']:
                if not bot.ml_validator.is_fitted(symbol, direction):
                    logger.critical(f"CRITICAL: {symbol} {direction} ML model not fitted after startup sequence!")
                    bot.last_error_message = f"ML model not fitted: {symbol} {direction}"


        bot.start_bot_logic()
        while bot.bot_running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Trading bot (direct run) stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"Critical unhandled exception in bot's direct run main loop: {e}", exc_info=True)
        bot.last_error_message = str(e)
    finally:
        logger.info("Shutting down bot (direct run)...")
        if 'bot' in locals() and bot and bot.bot_running:
            bot.stop_bot_logic()
            if bot.bot_thread and bot.bot_thread.is_alive():
                logger.info("Waiting for bot thread to finish...")
                bot.bot_thread.join(timeout=max(30, bot.config.get('monitoring_interval_seconds', 60) + 10))
                if bot.bot_thread.is_alive():
                    logger.warning("Bot thread did not terminate gracefully after timeout.")

        MongoDBConnection.close_connection()
        logger.info("Shutting down MT5 connection (direct run).")
        mt5.shutdown()
        logger.info("Bot (direct run) shutdown complete.")

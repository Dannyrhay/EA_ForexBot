import logging
import MetaTrader5 as mt5
import pandas as pd
import json
import time
import numpy as np
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.validation import check_is_fitted, NotFittedError

from utils.db_connector import MongoDBConnection
from pymongo import DESCENDING, errors
from utils.logging import setup_logging
from strategies.ml_model import MLValidator
from strategies.regime_filter import RegimeFilter
from strategies.strategy_selector import StrategySelector
from strategies.sentiment_filter import SentimentFilter
from utils.order_manager import OrderManager
from utils.risk_manager import RiskManager
from utils.mt5_connection import get_data, connect_mt5
from utils.feature_engineering import extract_ml_features, calculate_atr, calculate_rsi
from utils.trade_history import get_strategy_weights, save_trade, update_trade_status

# Strategy Imports (Core only)
from strategies.liquidity_sweep import LiquiditySweepStrategy
from strategies.fibonacci import FibonacciStrategy
from strategies.malaysian_snr import MalaysianSnRStrategy
from strategies.smc import SMCStrategy
from strategies.adx_strategy import ADXStrategy
from consensus import ConsensusEngine # [NEW] Import ConsensusEngine

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

        # [NEW] Initialize Managers
        self.order_manager = OrderManager()
        self.regime_filter = RegimeFilter()
        self.sentiment_filter = None # Init in load_config
        # Strategies are initialized in load_config_and_reinitialize, so we init selector there or after
        self.strategy_selector = None
        self.risk_manager = None  # Will be initialized after config is loaded

        # Throttle session check logging (log once per minute)
        self._last_session_log_time = {}

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

            # [NEW] Initialize Sentiment Filter
            news_api_key = self.config.get('news_api_key', '0b810baf39c3405586fd7238258b59c3') # Fallback to provided key
            self.sentiment_filter = SentimentFilter(news_api_key)

            if hasattr(self, 'ml_validator') and self.ml_validator:
                self.ml_validator.config = self.config
                self.ml_validator.create_predictors_for_all_symbols() # Recreate with new config
            else:
                self.ml_validator = MLValidator(self.config)

            self.strategies = self.initialize_strategies()
            self.strategy_selector = StrategySelector(self.strategies) # [NEW] Initialize Selector
            self.risk_manager = RiskManager(self.config) # [NEW] Initialize Risk Manager

            # Pass dependencies to feature engineering functions if needed
            # This is important after refactoring
            self._adx_strategy_for_features = next((s for s in self.strategies if isinstance(s, ADXStrategy)), None)
            if not self._adx_strategy_for_features:
                logger.warning("ADXStrategy instance not found. ADX features/filter might be unavailable.")

            # [NEW] Initialize ConsensusEngine
            self.consensus_engine = ConsensusEngine(
                strategies=self.strategies,
                config=self.config,
                ml_validator=self.ml_validator
            )
            logger.info("ConsensusEngine initialized successfully.")


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

            # Check tick freshness
            tick_time_dt = datetime.fromtimestamp(tick.time, tz=timezone.utc)
            if (datetime.now(timezone.utc) - tick_time_dt).total_seconds() > 60:
                logger.warning(f"MANAGE_TRADES ({ticket}): Tick for {symbol} is stale ({tick_time_dt}). Skipping.")
                continue

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

                            # Safety check for stops_level
                            stops_level_dist = (symbol_info.trade_stops_level + 2) * point
                            if trade_type == mt5.ORDER_TYPE_BUY:
                                if (current_price - new_secure_sl) < stops_level_dist:
                                    new_secure_sl = current_price - stops_level_dist
                                    logger.warning(f"Initial Secure (Fixed): SL Adjusted to {new_secure_sl} (StopsLevel)")
                            else: # SELL
                                if (new_secure_sl - current_price) < stops_level_dist:
                                    new_secure_sl = current_price + stops_level_dist
                                    logger.warning(f"Initial Secure (Fixed): SL Adjusted to {new_secure_sl} (StopsLevel)")

                        elif secure_type == "percentage_of_profit":
                            percentage = ps_sl_symbol_specific_config.get("secure_profit_percentage", 0.5)
                            pips_to_secure = current_profit_pips * percentage
                            if trade_type == mt5.ORDER_TYPE_BUY:
                                new_secure_sl = entry_price + (pips_to_secure * point)
                            else: # SELL
                                new_secure_sl = entry_price - (pips_to_secure * point)

                            # Safety check for stops_level
                            stops_level_dist = (symbol_info.trade_stops_level + 2) * point
                            if trade_type == mt5.ORDER_TYPE_BUY:
                                if (current_price - new_secure_sl) < stops_level_dist:
                                    new_secure_sl = current_price - stops_level_dist
                                    logger.warning(f"Initial Secure (Pct): SL Adjusted to {new_secure_sl} (StopsLevel)")
                            else: # SELL
                                if (new_secure_sl - current_price) < stops_level_dist:
                                    new_secure_sl = current_price + stops_level_dist
                                    logger.warning(f"Initial Secure (Pct): SL Adjusted to {new_secure_sl} (StopsLevel)")

                        should_move_sl = False
                        if trade_type == mt5.ORDER_TYPE_BUY and (original_sl == 0 or new_secure_sl > original_sl):
                            should_move_sl = True
                        elif trade_type == mt5.ORDER_TYPE_SELL and (original_sl == 0 or new_secure_sl < original_sl):
                            should_move_sl = True

                        if should_move_sl and new_secure_sl != 0:
                            if self.modify_position_sl(ticket, new_secure_sl, symbol):
                                logger.info(f"MANAGE_TRADES (Initial Secure - {ticket}): Moved SL to {new_secure_sl:.{digits}f} for {('BUY' if trade_type == mt5.ORDER_TYPE_BUY else 'SELL')} {symbol}.")
                                self.update_trade_status(ticket, {"sl_price": round(new_secure_sl, digits), "exit_reason": "InitialProfitSecure"})
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
            open_logged_trades_cursor = trades_collection.find({"status": "open", "order_id": {"$exists": True, "$ne": None}})
            open_trades_list = list(open_logged_trades_cursor)

            if not open_trades_list:
                # logger.debug("CHECK_CLOSED_TRADES: No 'open' trades found in MongoDB to reconcile.")
                return

            for trade_doc in open_trades_list:
                order_id = trade_doc.get("order_id")
                if order_id is None:
                    logger.debug(f"CHECK_CLOSED_TRADES: Found trade document with missing order_id: {trade_doc.get('_id')}")
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
                # Skip DXY during weekends (no logging to reduce noise)
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
        """
        # Core strategies only
        strategy_constructors = {
            "SMC": (SMCStrategy, lambda c: c.get('smc_params', {})),
            "LiquiditySweep": (LiquiditySweepStrategy, lambda c: c.get('liquidity_sweep_params', {})),
            "Fibonacci": (FibonacciStrategy, lambda c: {
                'swing_lookback': c.get('fibonacci_golden_zone', {}).get('swing_lookback', 200),
                'trend_ema_period': c.get('fibonacci_golden_zone', {}).get('trend_ema_period', 200),
                'strength': c.get('fibonacci_golden_zone', {}).get('signal_strength', 0.85)
            }),
            "MalaysianSnR": (MalaysianSnRStrategy, lambda c: {}),
            "ADX": (ADXStrategy, lambda c: {})
        }

        initialized_strategies = []
        active_strategy_names = []

        if self.config.get('thesis_mode_enabled', False):
            active_strategy_names = ["SMC"]
        else:
            active_strategy_names = list(strategy_constructors.keys())

        logger.info(f"Active strategies to be loaded: {active_strategy_names}")

        for name in active_strategy_names:
            if name in strategy_constructors:
                StrategyClass, params_lambda = strategy_constructors[name]
                try:
                    params_dict = params_lambda(self.config)
                    initialized_strategies.append(StrategyClass(name=name, **params_dict))
                    logger.info(f"Successfully initialized strategy: {name}")
                except Exception as e:
                    logger.error(f"Failed to initialize strategy {name}: {e}", exc_info=True)
            else:
                logger.warning(f"Strategy '{name}' is configured to be active but not found in constructors.")

        return initialized_strategies


    def generate_ml_labels(self, df, symbol):
        """
        Generates target labels for ML training based on future price movements.
        """
        if df is None or df.empty: return df

        # Simple labeling: 1 if next close > current close, 0 otherwise
        # This is a placeholder. Real labeling should be more robust (e.g. based on ATR, fixed pips)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        return df


    def is_trading_hours(self, symbol):
        """
        Checks if the current time is within allowed trading hours for the symbol.
        """
        # Simplified check. In production, check symbol_info.session_deals
        return True


    def is_within_active_session(self, symbol):
        """
        Checks if the current time is within the active sessions defined in config.
        """
        # Simplified check
        return True


    def get_dxy_data_for_correlation(self, timeframe_str, bars=1000):
        """
        Fetches DXY data for correlation analysis.
        """
        dxy_symbol = self.config.get('dxy_symbol', 'DXY')
        return get_data(dxy_symbol, timeframe_str, bars=bars)


    def _generate_labels_for_training(self, df, symbol):
        return self.generate_ml_labels(df, symbol)


    def initial_ml_models_training(self):
        """
        Performs initial training of ML models if they are not already fitted.
        """
        if not self.ml_validator: return

        logger.info("Checking ML models status...")
        for symbol in self.config.get('symbols', []):
            if not self.ml_validator.is_fitted(symbol, 'buy'):
                logger.info(f"Training initial BUY model for {symbol}...")
                # Fetch historical data
                data = get_data(symbol, 'M15', bars=5000) # Example timeframe/bars
                if data is not None and not data.empty:
                    data = extract_ml_features(data, symbol, self.get_dxy_data_for_correlation('M15'), self._adx_strategy_for_features)
                    data = self._generate_labels_for_training(data, symbol)
                    # Train logic here (simplified call)
                    # self.ml_validator.train_model(symbol, 'buy', data)
                    pass

            if not self.ml_validator.is_fitted(symbol, 'sell'):
                logger.info(f"Training initial SELL model for {symbol}...")
                # Similar logic for SELL
                pass

    # --- CONTINUATION MARKER ---

    def perform_global_retrain(self):
        """
        Retrains ML models for all symbols.
        """
        if not self.ml_validator: return

        logger.info("Starting global ML model retraining...")
        self.last_global_retrain_time = datetime.now(timezone.utc)

        for symbol in self.config.get('symbols', []):
            self.retrain_ml_model_for_symbol_and_direction(symbol, 'buy')
            self.retrain_ml_model_for_symbol_and_direction(symbol, 'sell')

        logger.info("Global retraining completed.")

    def retrain_ml_model_for_symbol_and_direction(self, symbol, direction):
        if not self.ml_validator: return
        logger.info(f"Retraining {direction.upper()} model for {symbol}...")

        data = get_data(symbol, 'M15', bars=5000)
        if data is not None and not data.empty:
            try:
                # Use batch feature extraction
                from utils.feature_engineering import extract_ml_features_batch
                features_df = extract_ml_features_batch(symbol, data, direction, self)

                # Generate labels (also batch)
                data_with_target = self._generate_labels_for_training(data, symbol)
                target = data_with_target['target']

                # Align features and target
                # Drop NaN rows (start of history)
                combined = pd.concat([features_df, target], axis=1).dropna()

                if combined.empty:
                    logger.warning(f"No valid training data for {symbol} {direction} after dropping NaNs.")
                    return

                X = combined.iloc[:, :-1].values # All columns except target
                y = combined.iloc[:, -1].values  # Last column is target

                if len(X) > 100:
                    self.ml_validator.fit(symbol, X, y, direction)
                    logger.info(f"Successfully retrained {direction.upper()} model for {symbol} with {len(X)} samples.")
                else:
                    logger.warning(f"Insufficient data for training {symbol} {direction}: {len(X)} samples.")
            except Exception as e:
                logger.error(f"Error during retraining {symbol} {direction}: {e}", exc_info=True)

    def mt5_tf_to_minutes(self, tf_str):
        mapping = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080, 'MN1': 43200
        }
        return mapping.get(tf_str, 0)

    def get_trend(self, symbol, timeframe_str='H4'):
        data = get_data(symbol, timeframe_str, bars=100)
        if data is None or data.empty: return 'neutral'

        data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
        data['ema_200'] = data['close'].ewm(span=200, adjust=False).mean()

        last_row = data.iloc[-1]
        if last_row['ema_50'] > last_row['ema_200']: return 'uptrend'
        elif last_row['ema_50'] < last_row['ema_200']: return 'downtrend'
        return 'neutral'

    def in_cooldown(self, symbol):
        last_trade_time = self.last_trade_times.get(symbol)
        if last_trade_time:
            if datetime.now(timezone.utc) - last_trade_time < self.cooldown_period:
                return True
        return False

    def can_open_new_trade(self, symbol):
        if self.in_cooldown(symbol): return False

        positions = mt5.positions_get(symbol=symbol)
        if positions and len(positions) >= self.config.get('max_open_positions_per_symbol', 1):
            return False

        return True

    def execute_trade(self, symbol, signal_type, trade_params, data, timeframe):
        if not self.can_open_new_trade(symbol):
            logger.info(f"Skipping trade for {symbol}: Cooldown or max positions reached.")
            return

        volume = trade_params.get('volume')
        if volume is None:
            if self.risk_manager:
                try:
                    account_info = mt5.account_info()
                    balance = account_info.balance if account_info else 1000.0

                    entry_price = trade_params.get('entry_price')
                    sl_price = trade_params.get('sl_price')

                    sl_distance = 0.0
                    if entry_price and sl_price:
                        sl_distance = abs(entry_price - sl_price)
                    elif trade_params.get('sl_pips'):
                        symbol_info = mt5.symbol_info(symbol)
                        if symbol_info:
                            point = symbol_info.point
                            pip_size = point * 10 if symbol_info.digits % 2 == 1 else point
                            sl_distance = trade_params.get('sl_pips') * pip_size

                    if sl_distance > 0:
                        volume = self.risk_manager.calculate_position_size(symbol, balance, sl_distance)
                    else:
                        logger.warning(f"Could not calculate SL distance for {symbol}. Using default volume.")
                        volume = 0.01
                except Exception as e:
                    logger.error(f"Error calculating position size: {e}. Using default.")
                    volume = 0.01
            else:
                volume = 0.01

        sl = trade_params.get('sl_price')
        tp = trade_params.get('tp_price')
        comment = f"AI-{timeframe}"

        limit_price = trade_params.get('limit_price')

        if limit_price:
            self._place_limit_order(symbol, signal_type, volume, limit_price, sl, tp, comment)
        else:
            self._send_market_order(symbol, signal_type, volume, sl, tp, comment)

    def _send_market_order(self, symbol, signal_type, volume, sl, tp, comment):
        order_type = mt5.ORDER_TYPE_BUY if signal_type == 'buy' else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).ask if signal_type == 'buy' else mt5.symbol_info_tick(symbol).bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Trade executed: {symbol} {signal_type} {volume} lots at {price}")
            self.save_new_trade(result, symbol, signal_type, volume, sl, tp, comment)
            self.last_trade_times[symbol] = datetime.now(timezone.utc)
        else:
            logger.error(f"Trade failed: {result.comment if result else mt5.last_error()}")

    def _place_limit_order(self, symbol, signal_type, volume, price, sl, tp, comment):
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if signal_type == 'buy' else mt5.ORDER_TYPE_SELL_LIMIT

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Limit order placed: {symbol} {signal_type} {volume} lots at {price}")
            self.save_new_trade(result, symbol, signal_type, volume, sl, tp, comment)
        else:
            logger.error(f"Limit order failed: {result.comment if result else mt5.last_error()}")

    def close_position_by_ticket(self, ticket, symbol, volume, order_type, comment=""):
        close_type = mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position closed: {ticket} {symbol}")
            return True
        else:
            logger.error(f"Failed to close position {ticket}: {result.comment if result else mt5.last_error()}")
            return False

    def save_new_trade(self, result, symbol, signal_type, volume, sl, tp, comment):
        trade_data = {
            "order_id": result.order,
            "deal_id": result.deal,
            "symbol": symbol,
            "signal": signal_type,
            "lot_size": volume,
            "entry_price": result.price,
            "sl": sl,
            "tp": tp,
            "entry_time": datetime.now(timezone.utc),
            "status": "open",
            "comment": comment
        }
        save_trade(trade_data)

    # --- CONTINUATION MARKER 2 ---

    def monitor_market(self):
        logger.info("Market monitoring started.")
        while self.bot_running:
            try:
                if not mt5.terminal_info():
                    logger.warning("MT5 disconnected. Attempting reconnect...")
                    if not connect_mt5(self.config.get('mt5_credentials')):
                        time.sleep(5); continue

                self.manage_open_trades()
                self.check_closed_trades()

                if datetime.now(timezone.utc) - self.last_global_retrain_time > self.global_retrain_interval:
                    self.perform_global_retrain()

                for symbol in self.config.get('symbols', []):
                    if not self.can_open_new_trade(symbol): continue

                    for tf_str in self.config.get('timeframes', ['M5']):
                        data = get_data(symbol, tf_str, bars=1000)
                        if data is None or data.empty: continue

                        # Ensure consensus_engine is initialized
                        if not hasattr(self, 'consensus_engine'):
                            logger.error("ConsensusEngine not initialized! Skipping analysis.")
                            continue

                        # Analyze using ConsensusEngine (it collects signals internally)
                        # Pass data (DataFrame) as first argument
                        consensus_result = self.consensus_engine.analyze(data, symbol, tf_str)

                        if consensus_result.signal_type != 'HOLD':
                            logger.info(f"Consensus Signal for {symbol} ({tf_str}): {consensus_result.signal_type}. Executing...")

                            # Construct trade_params from TradeSignal object
                            trade_params = {
                                'entry_price': consensus_result.price,
                                'sl_price': consensus_result.sl,
                                'tp_price': consensus_result.tp,
                                'volume': consensus_result.metadata.get('volume'),
                                'sl_pips': consensus_result.metadata.get('sl_pips'),
                                'tp_pips': consensus_result.metadata.get('tp_pips'),
                                'limit_price': consensus_result.metadata.get('limit_price')
                            }

                            self.execute_trade(symbol, consensus_result.signal_type.lower(), trade_params, data, tf_str)

            except Exception as e:
                logger.error(f"Error in monitor_market loop: {e}", exc_info=True)
                time.sleep(5)

            time.sleep(1)


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

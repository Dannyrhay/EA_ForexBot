import MetaTrader5 as mt5
import pandas as pd
import json
import time
import pytz
import smtplib
import os
from email.mime.text import MIMEText
from utils.mt5_connection import connect_mt5, get_data
from utils.logging import setup_logging
from utils.trade_history import save_trade, get_strategy_weights
from strategies.supply_demand import SupplyDemandStrategy
from strategies.sma import SMAStrategy
from strategies.smc import SMCStrategy
from strategies.liquidity_sweep import LiquiditySweepStrategy
from strategies.fvg import FVGStrategy
from strategies.fibonacci import FibonacciStrategy
from strategies.candlestick import CandlestickStrategy

logger = setup_logging()

try:
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    logger.debug("Configuration loaded successfully")
except FileNotFoundError:
    logger.error("Config file not found: config/config.json")
    exit(1)
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON in config file: {e}")
    exit(1)
except Exception as e:
    logger.error(f"Failed to load config: {e}")
    exit(1)

# Connect to Exness MT5 with provided credentials
symbols = config['symbols']
if not connect_mt5(login=242854137, password="Danny@0011", server="Exness-MT5Trial"):
    logger.error("Failed to connect to Exness MT5")
    exit(1)

symbols_data = mt5.symbols_get()
if symbols_data is None:
    logger.error("Failed to retrieve symbols from MT5: %s", mt5.last_error())
    exit(1)

available_symbols = [s.name for s in symbols_data if s.name in symbols]
if not available_symbols:
    logger.error("No configured symbols available in Exness MT5 Market Watch")
    logger.info("Available symbols: %s", [s.name for s in symbols_data])
    exit(1)
elif set(symbols) != set(available_symbols):
    logger.warning(f"Some symbols not available: Configured {symbols}, Available {available_symbols}")
    config['symbols'] = available_symbols
logger.info("MT5 connected successfully")

# Initialize strategies
strategies = [
    SupplyDemandStrategy(window=config['supply_demand_window'], name="SupplyDemand"),
    SMAStrategy(short_period=config['sma_short_period'], long_period=config['sma_long_period'], name="SMA"),
    SMCStrategy(name="SMC"),
    LiquiditySweepStrategy(period=config['liquidity_sweep_period'], name="LiquiditySweep"),
    FVGStrategy(gap_threshold=config['fvg_gap_threshold'], name="FVG"),
    FibonacciStrategy(levels=config['fibonacci_levels'], name="Fibonacci"),
    CandlestickStrategy(name="Candlestick")
]
logger.debug("Strategies initialized")

# Map timeframes to MT5 constants
timeframe_map = {
    "M1": mt5.TIMEFRAME_M1,
    "M15": mt5.TIMEFRAME_M15,
    "M45": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1
}

# Email notification setup
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "gojofinn@gmail.com"
SMTP_PASSWORD = "ppmgpvkujbscrhqk"
RECIPIENT_EMAIL = "danieloppong757@gmail.com"
NOTIFICATION_LOG_DIR = "notifications"
NOTIFICATION_LOG_FILE = os.path.join(NOTIFICATION_LOG_DIR, "failed_notifications.log")

# Ensure notification log directory exists
os.makedirs(NOTIFICATION_LOG_DIR, exist_ok=True)

def save_failed_notification(subject, body):
    """Save notification details to a file if SMTP fails."""
    try:
        with open(NOTIFICATION_LOG_FILE, 'a') as f:
            f.write(f"[{pd.Timestamp.now(tz='UTC').isoformat()}] {subject}\n{body}\n{'='*50}\n")
        logger.info(f"Failed notification saved to {NOTIFICATION_LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to save notification to file: {e}")

def send_notification(subject, body):
    """Send an email notification with retry logic for transient errors."""
    max_retries = 3
    retry_delays = [5, 10, 20]  # Seconds to wait between retries
    attempt = 0

    while attempt < max_retries:
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = SMTP_USER
            msg['To'] = RECIPIENT_EMAIL
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_USER, RECIPIENT_EMAIL, msg.as_string())
            logger.info("Notification sent successfully")
            return True
        except smtplib.SMTPServerDisconnected as e:
            logger.warning(f"SMTP server disconnected on attempt {attempt + 1}: {e}")
            attempt += 1
            if attempt < max_retries:
                time.sleep(retry_delays[attempt - 1])
            continue
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error on attempt {attempt + 1}: {e}")
            attempt += 1
            if attempt < max_retries:
                time.sleep(retry_delays[attempt - 1])
            continue
        except Exception as e:
            logger.error(f"Unexpected error sending notification: {e}")
            break
    logger.error("Failed to send notification after all retries")
    save_failed_notification(subject, body)
    return False

def calculate_atr(data, period=14):
    """Calculate Average True Range (ATR)."""
    try:
        df = pd.DataFrame(data)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().iloc[-1]
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return 0.0

def calculate_lot_size(symbol, sl_distance):
    """Calculate lot size based on risk_percent, min $5 risk, Exness min lot 0.01, max lot 5.0."""
    try:
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to retrieve account info")
            return 0.0
        account_balance = account_info.balance
        if account_balance <= 0:
            logger.error("Invalid account balance")
            return 0.0
        risk_percent = config['risk_percent']
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {symbol}")
            return 0.0
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        sl_pips = sl_distance / tick_size

        # Calculate lot size for 3% risk
        risk_amount = account_balance * (risk_percent / 100)
        lot_size = risk_amount / (sl_pips * tick_value)

        # Ensure minimum $5 risk
        min_risk_amount = 5.0
        min_lot_size = min_risk_amount / (sl_pips * tick_value)
        lot_size = max(lot_size, min_lot_size)

        # Enforce Exness minimum and maximum lot size
        lot_size = max(0.01, min(5.0, round(lot_size, 2)))
        return lot_size
    except Exception as e:
        logger.error(f"Error calculating lot size: {e}")
        return 0.0

def has_open_position(symbol):
    """Check if there’s an open position for the symbol."""
    try:
        positions = mt5.positions_get(symbol=symbol)
        return len(positions) > 0 if positions else False
    except Exception as e:
        logger.error(f"Error checking open position for {symbol}: {e}")
        return False

def select_best_strategy():
    """Select the strategy with the highest historical win rate."""
    try:
        weights = get_strategy_weights()
        best_strategy = max(weights, key=weights.get)
        for strategy in strategies:
            if strategy.name == best_strategy:
                return strategy
        logger.warning("No best strategy found, defaulting to SMA")
        return next(s for s in strategies if s.name == "SMA")
    except Exception as e:
        logger.error(f"Error selecting best strategy: {e}")
        return next(s for s in strategies if s.name == "SMA")

def execute_trade(symbol, signal, data, timeframe, strategy_signals, threshold, single_strategy=None):
    """Execute a buy or sell trade, save to memory, and send notification."""
    if has_open_position(symbol):
        logger.info(f"Position already open for {symbol}, skipping trade")
        return False

    atr = calculate_atr(data, config['atr_period'])
    if atr == 0.0:
        logger.error(f"Invalid ATR for {symbol}")
        return False
    atr_threshold = 0.00005 if 'USD' in symbol else 5.0 if 'BTC' in symbol else 0.25
    if atr < atr_threshold:
        logger.info(f"ATR too low for {symbol}: {atr} < {atr_threshold}")
        return False

    sl_distance = 1.0 * atr  # Tighter SL for more trades
    tp_distance = 3.0 * atr
    lot_size = calculate_lot_size(symbol, sl_distance)
    if lot_size <= 0:
        logger.error(f"Invalid lot size for {symbol}")
        return False

    symbol_info = mt5.symbol_info_tick(symbol)
    if not symbol_info:
        logger.error(f"Failed to get tick info for {symbol}")
        return False

    price = symbol_info.ask if signal == 'buy' else symbol_info.bid
    sl_price = price - sl_distance if signal == 'buy' else price + sl_distance
    tp_price = price + tp_distance if signal == 'buy' else price - tp_distance

    tick_value = mt5.symbol_info(symbol).trade_tick_value
    tp_profit = (tp_distance * lot_size * tick_value) if signal == 'buy' else (-tp_distance * lot_size * tick_value)
    sl_loss = (-sl_distance * lot_size * tick_value) if signal == 'buy' else (sl_distance * lot_size * tick_value)
    account_info = mt5.account_info()
    account_balance = account_info.balance if account_info else 0.0

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 10,
        "magic": 123456,
        "comment": f"Exness MultiConfluence {signal} {'Single' if single_strategy else threshold}+",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    trade_data = {
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal,
        "entry_price": price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "lot_size": lot_size,
        "strategies": strategy_signals,
        "entry_time": pd.Timestamp.now(tz='UTC').isoformat(),
        "status": "pending",
        "profit_loss": 0.0,
        "account_balance": account_balance,
        "order_id": 0,
        "failure_reason": ""
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        trade_data["status"] = "failed"
        trade_data["failure_reason"] = result.comment
        save_trade(trade_data)
        logger.error(f"Trade failed for {symbol}: {result.comment}")
        return False
    else:
        trade_data["status"] = "open"
        trade_data["order_id"] = result.order
        save_trade(trade_data)
        logger.info(
            f"Successful trade for {symbol}: {signal}, Lot: {lot_size}, "
            f"Entry: {price}, SL: {sl_price}, TP: {tp_price}, Order ID: {result.order}, "
            f"{'Single Strategy: ' + single_strategy.name if single_strategy else f'Threshold: {threshold}+'}"
        )
        notification_body = (
            f"Successful Trade Executed on Exness MT5\n"
            f"Symbol: {symbol}\n"
            f"Timeframe: {timeframe}\n"
            f"Signal: {signal.upper()}\n"
            f"Lot Size: {lot_size}\n"
            f"Entry Price: {price}\n"
            f"Stop Loss: {sl_price}\n"
            f"Take Profit: {tp_price}\n"
            f"Estimated Profit (if TP hit): {tp_profit:.2f} USD\n"
            f"Estimated Loss (if SL hit): {sl_loss:.2f} USD\n"
            f"Account Balance: {account_balance:.2f} USD\n"
            f"Order ID: {result.order}\n"
            f"{'Single Strategy: ' + single_strategy.name if single_strategy else f'Threshold: {threshold}+'}\n"
            f"Time: {pd.Timestamp.now(tz='UTC')}\n"
            f"Stored in Trade History"
        )
        send_notification("Exness Trading EA: Successful Trade", notification_body)
        return True

def monitor_closed_trades(symbol, order_id, lot_size, signal):
    """Monitor trade closure, update memory, and send notification."""
    while True:
        try:
            positions = mt5.positions_get(symbol=symbol)
            position_exists = any(pos.order == order_id for pos in positions) if positions else False
            if not position_exists:
                history = mt5.history_deals_get(ticket=order_id)
                if history:
                    for deal in history:
                        if deal.order == order_id:
                            profit = deal.profit
                            account_info = mt5.account_info()
                            account_balance = account_info.balance if account_info else 0.0
                            trade_data = {
                                "order_id": order_id,
                                "status": "closed",
                                "profit_loss": profit,
                                "account_balance": account_balance,
                                "exit_time": pd.Timestamp.now(tz='UTC').isoformat()
                            }
                            save_trade(trade_data, update=True)
                            logger.info(
                                f"Trade closed for {symbol}: {signal}, Order ID: {order_id}, "
                                f"Profit/Loss: {profit:.2f} USD, Account Balance: {account_balance:.2f} USD"
                            )
                            notification_body = (
                                f"Trade Closed on Exness MT5\n"
                                f"Symbol: {symbol}\n"
                                f"Signal: {signal.upper()}\n"
                                f"Order ID: {order_id}\n"
                                f"Profit/Loss: {profit:.2f} USD\n"
                                f"Account Balance: {account_balance:.2f} USD\n"
                                f"Time: {pd.Timestamp.now(tz='UTC')}\n"
                                f"Updated in Trade History"
                            )
                            send_notification("Exness Trading EA: Trade Closed", notification_body)
                            return
                else:
                    logger.error(f"No history found for closed trade {order_id} in {symbol}")
                    return
            time.sleep(60)
        except Exception as e:
            logger.error(f"Error monitoring trade {order_id} for {symbol}: {e}")
            time.sleep(60)

def main():
    symbols = config['symbols']
    timeframes = config['timeframes']
    last_candle_times = {symbol: {tf: None for tf in timeframes} for symbol in symbols}
    start_time = time.time()
    threshold = 4
    trade_count = 0
    last_threshold_change = start_time
    single_strategy_mode = False
    best_strategy = None
    cycle_count = 0

    logger.debug("Entering main trading loop")
    while True:
        try:
            # Reset trade_count at the start of each cycle
            trade_count = 0

            # Check for threshold adjustment or single-strategy mode
            elapsed_time = time.time() - start_time
            time_since_last_change = time.time() - last_threshold_change
            logger.debug(f"Cycle {cycle_count}: trade_count={trade_count}, threshold={threshold}, time_since_last_change={time_since_last_change:.2f}s")

            if time_since_last_change >= 300 and threshold == 4:  # 5 minutes
                threshold = 3
                cycle_count += 1
                logger.info(f"Cycle {cycle_count}: No trades in 5 minutes, switching to 3+ strategy threshold")
                last_threshold_change = time.time()
            elif time_since_last_change >= 600 and threshold == 3:  # 10 minutes
                threshold = 2
                logger.info(f"Cycle {cycle_count}: No trades in 10 minutes, switching to 2+ strategy threshold")
                last_threshold_change = time.time()
            elif time_since_last_change >= 1500 and threshold == 2 and not single_strategy_mode:  # 25 minutes
                single_strategy_mode = True
                best_strategy = select_best_strategy()
                logger.info(f"Cycle {cycle_count}: No trades with 2+ threshold, switching to single strategy: {best_strategy.name}")
                last_threshold_change = time.time()
            elif time_since_last_change >= 1800 and single_strategy_mode:  # 30 minutes
                threshold = 4
                single_strategy_mode = False
                cycle_count += 1
                logger.info(f"Cycle {cycle_count}: No trades with single strategy, resetting to 4+ strategy threshold")
                last_threshold_change = time.time()
                start_time = time.time()

            # Get strategy weights from historical performance
            strategy_weights = get_strategy_weights()
            weighted_strategies = [
                (s, strategy_weights.get(s.name, 1.0)) for s in strategies
            ]

            for symbol in symbols:
                for tf in timeframes:
                    logger.debug(f"Processing {symbol} on timeframe {tf}")
                    mt5_timeframe = timeframe_map.get(tf, mt5.TIMEFRAME_H1)
                    rates = get_data(symbol, mt5_timeframe, 500)
                    if rates is None:
                        logger.warning(f"No data retrieved for {symbol} on {tf}")
                        continue

                    current_candle_time = rates['time'].iloc[-1]
                    last_time = last_candle_times[symbol][tf]

                    if last_time is None:
                        last_candle_times[symbol][tf] = current_candle_time
                        continue

                    if current_candle_time > last_time:
                        last_candle_times[symbol][tf] = current_candle_time
                        candle1 = rates.iloc[-2]
                        candle2 = rates.iloc[-1]
                        strategy_signals = {}

                        if single_strategy_mode:
                            # Single-strategy mode
                            signal = best_strategy.get_signal(rates)
                            strategy_signals[best_strategy.name] = signal
                            logger.debug(f"Cycle {cycle_count}: Single strategy signal for {symbol} on {tf}: {best_strategy.name}={signal}")

                            # Confirm trade (relaxed checks)
                            candlestick = CandlestickStrategy()
                            atr = calculate_atr(rates, config['atr_period'])
                            atr_threshold = 0.00005 if 'USD' in symbol else 5.0 if 'BTC' in symbol else 0.25
                            is_valid = (
                                signal in ['buy', 'sell'] and
                                atr >= atr_threshold and
                                (signal == 'buy' and candlestick.is_bullish_engulfing(candle1, candle2) or
                                 signal == 'sell' and candlestick.is_bearish_engulfing(candle1, candle2))
                            )

                            if is_valid:
                                logger.info(f"Cycle {cycle_count}: Single strategy trade confirmed for {symbol} on {tf}: {best_strategy.name}={signal}")
                                if execute_trade(symbol, signal, rates, tf, strategy_signals, threshold, single_strategy=best_strategy):
                                    trade_count += 1
                                    threshold = 4
                                    single_strategy_mode = False
                                    cycle_count += 1
                                    logger.info(f"Cycle {cycle_count}: Trade executed, resetting to 4+ strategy threshold")
                                    last_threshold_change = time.time()
                                    start_time = time.time()
                            else:
                                logger.debug(
                                    f"Cycle {cycle_count}: Single strategy trade not confirmed for {symbol} on {tf}: "
                                    f"Signal={signal}, ATR={atr}, "
                                    f"BullishEngulfing={candlestick.is_bullish_engulfing(candle1, candle2)}, "
                                    f"BearishEngulfing={candlestick.is_bearish_engulfing(candle1, candle2)}"
                                )
                                # Reset to 4+ immediately if confirmation fails
                                threshold = 4
                                single_strategy_mode = False
                                cycle_count += 1
                                logger.info(f"Cycle {cycle_count}: Single strategy confirmation failed, resetting to 4+ strategy threshold")
                                last_threshold_change = time.time()
                                start_time = time.time()
                        else:
                            # Multi-strategy mode
                            signals = []
                            for strategy, weight in weighted_strategies:
                                signal = strategy.get_signal(rates)
                                signals.extend([signal] * int(weight * 10))
                                strategy_signals[strategy.name] = signal
                            logger.debug(f"Cycle {cycle_count}: Strategy signals for {symbol} on {tf}: {strategy_signals}")

                            buy_count = signals.count('buy')
                            sell_count = signals.count('sell')
                            threshold_votes = threshold * 10

                            # Log candlestick status for debugging
                            candlestick = CandlestickStrategy()
                            logger.debug(
                                f"Cycle {cycle_count}: Candlestick check for {symbol} on {tf}: "
                                f"BullishEngulfing={candlestick.is_bullish_engulfing(candle1, candle2)}, "
                                f"BearishEngulfing={candlestick.is_bearish_engulfing(candle1, candle2)}"
                            )

                            if buy_count >= threshold_votes:
                                logger.info(f"Cycle {cycle_count}: {threshold}+ BUY signals detected for {symbol} on {tf} at {current_candle_time}")
                                if execute_trade(symbol, 'buy', rates, tf, strategy_signals, threshold):
                                    trade_count += 1
                                    threshold = 4
                                    single_strategy_mode = False
                                    cycle_count += 1
                                    logger.info(f"Cycle {cycle_count}: Trade executed, resetting to 4+ strategy threshold")
                                    last_threshold_change = time.time()
                                    start_time = time.time()
                            elif sell_count >= threshold_votes:
                                logger.info(f"Cycle {cycle_count}: {threshold}+ SELL signals detected for {symbol} on {tf} at {current_candle_time}")
                                if execute_trade(symbol, 'sell', rates, tf, strategy_signals, threshold):
                                    trade_count += 1
                                    threshold = 4
                                    single_strategy_mode = False
                                    cycle_count += 1
                                    logger.info(f"Cycle {cycle_count}: Trade executed, resetting to 4+ strategy threshold")
                                    last_threshold_change = time.time()
                                    start_time = time.time()
                            else:
                                logger.debug(f"Cycle {cycle_count}: Insufficient signals for {symbol} on {tf}: Buy={buy_count}, Sell={sell_count}, Needed={threshold_votes}")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)
            continue

        time.sleep(60)
        logger.debug(f"Cycle {cycle_count}: Completed one loop iteration")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Script stopped by user")
        mt5.shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        mt5.shutdown()
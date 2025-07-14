import MetaTrader5 as mt5
import pandas as pd
from utils.logging import setup_logging
import os
from .data_cache import data_cache_instance
from datetime import datetime

logger = setup_logging()

TIMEFRAME_MAP = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
    'W1': mt5.TIMEFRAME_W1,
    'MN1': mt5.TIMEFRAME_MN1,
}

def connect_mt5(config_credentials=None):
    """
    Initialize connection to MetaTrader 5 terminal.
    Uses environment variables as primary, falls back to config_credentials if provided.

    Args:
        config_credentials (dict, optional): Credentials from the config file.
                                            Example: {'login': ID, 'password': 'PASS', 'server': 'SERVER', 'mt5_terminal_path': 'PATH'}

    Returns:
        bool: True if connection successful, False otherwise
    """
    login_id_env = os.environ.get('MT5_LOGIN')
    password_env = os.environ.get('MT5_PASSWORD')
    server_env = os.environ.get('MT5_SERVER')
    path_env = os.environ.get('MT5_TERMINAL_PATH')

    login_id = None
    password = None
    server = None
    path = None

    if login_id_env and password_env and server_env:
        logger.info("Using MT5 credentials from environment variables.")
        login_id = login_id_env
        password = password_env
        server = server_env
        path = path_env
    elif config_credentials:
        logger.info("Using MT5 credentials from configuration file.")
        login_id = config_credentials.get('login')
        password = config_credentials.get('password')
        server = config_credentials.get('server')
        path = config_credentials.get('mt5_terminal_path')
    else:
        logger.error("MT5 credentials not found in environment variables or config.")
        return False

    if not (login_id and password and server):
        logger.error("Essential MT5 credentials (login, password, server) are missing.")
        return False

    try:
        login_id = int(login_id)
    except (ValueError, TypeError):
        logger.error(f"MT5_LOGIN or config login must be an integer. Got: {login_id}")
        return False

    mt5_path_arg_to_pass = path if path and path.strip() else ""

    logger.info(f"Attempting to initialize MT5 with Login ID: {login_id}, Server: {server}, Path: '{mt5_path_arg_to_pass}'")

    try:
        if not mt5.initialize(login=login_id, password=password, server=server, path=mt5_path_arg_to_pass):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            terminal_info = mt5.terminal_info()
            if terminal_info:
                logger.error(f"MT5 Terminal Info: Name: {terminal_info.name}, Company: {terminal_info.company}, Path: {terminal_info.path}")
            else:
                logger.error("Could not retrieve MT5 terminal info after failed initialization.")
            return False

        logger.info("MT5 connection established successfully.")
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"Account Info: Login: {account_info.login}, Name: {account_info.name}, Server: {account_info.server}, Balance: {account_info.balance} {account_info.currency}")
        else:
            logger.warning("Could not retrieve account info after MT5 connection.")
        return True
    except Exception as e:
        logger.error(f"An unexpected error occurred while connecting to MT5: {e}", exc_info=True)
        return False

def get_data(symbol, timeframe_str, bars=None, start_time=None, from_date=None, to_date=None, max_bars_to_store=50000):
    """
    Fetch historical data for a given symbol and timeframe, using cache or direct fetch.
    Handles fetching by bars, start_time, or a date range.

    Args:
        symbol (str): Trading symbol (e.g., 'XAUUSDm')
        timeframe_str (str): Timeframe string (e.g., 'M5')
        bars (int, optional): Number of bars to fetch. Required for caching or start_time fetch.
        start_time (datetime, optional): Start time for data fetching using copy_rates_from.
        from_date (datetime, optional): The start date for fetching data with copy_rates_range.
        to_date (datetime, optional): The end date for fetching data with copy_rates_range.
        max_bars_to_store (int): Maximum bars to store in cache.

    Returns:
        pd.DataFrame: Historical data with 'time' as a column or None if failed.
    """
    timeframe = TIMEFRAME_MAP.get(timeframe_str)
    if timeframe is None:
        logger.error(f"Invalid timeframe_str: {timeframe_str}")
        return None

    try:
        if not mt5.terminal_info():
            logger.warning("MT5 not initialized. Attempting to connect.")
            # Assuming connect_mt5 uses a global config or env vars correctly
            if not connect_mt5():
                logger.error(f"MT5 connection failed for {symbol}.")
                return None

        symbol_info_check = mt5.symbol_info(symbol)
        if not symbol_info_check:
            logger.error(f"Symbol {symbol} not found. MT5 Last Error: {mt5.last_error()}")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}.")
                return None
            else:
                logger.info(f"Symbol {symbol} selected. Retrying fetch.")

        # --- MODIFIED LOGIC ---
        # Prioritize fetching by date range if from_date and to_date are provided
        if from_date is not None and to_date is not None:
            logger.debug(f"Fetching {symbol} on {timeframe_str} from {from_date} to {to_date}")
            rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)
            if rates is None or len(rates) == 0:
                logger.warning(f"No data fetched for {symbol} on {timeframe_str} using date range. Error: {mt5.last_error()}")
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
            if not all(col in df.columns for col in required_columns):
                 logger.error(f"Missing columns in fetched data for {symbol}: {df.columns}")
                 return None
            logger.info(f"Fetched {len(df)} bars for {symbol} on {timeframe_str} using date range.")
            return df[required_columns]

        # Check if 'bars' argument is provided for other methods
        if bars is None:
            logger.error("get_data requires the 'bars' argument when not using from_date and to_date.")
            return None

        # Fetch using start_time (no cache)
        if start_time is not None:
            rates = mt5.copy_rates_from(symbol, timeframe, start_time, bars)
            if rates is None or len(rates) == 0:
                logger.error(f"No data fetched for {symbol} on {timeframe_str} from start_time: {mt5.last_error()}")
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing columns in data for {symbol}: {df.columns}")
                return None
            logger.info(f"Fetched {len(df)} bars for {symbol} on {timeframe_str} (direct from start_time)")
            return df[required_columns]

        # Default to caching logic
        else:
            cached_df, last_bar_ts = data_cache_instance.get_cached_ohlc(symbol, timeframe_str)
            if cached_df is None or len(cached_df) < bars:
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, max(bars, max_bars_to_store))
                if rates is None or len(rates) == 0:
                    logger.error(f"No initial data fetched for {symbol} on {timeframe_str} for cache")
                    return None
                df_new = pd.DataFrame(rates)
                df_new['time'] = pd.to_datetime(df_new['time'], unit='s')
                data_cache_instance.update_ohlc_cache(symbol, timeframe_str, df_new, max_bars_to_store)
                cached_df, _ = data_cache_instance.get_cached_ohlc(symbol, timeframe_str)
                logger.info(f"Fetched {len(df_new)} initial bars for {symbol} on {timeframe_str} for cache")
            else:
                last_bar_time = cached_df['time'].iloc[-1]
                date_from_update = last_bar_time + pd.Timedelta(seconds=1)
                date_to_update = datetime.utcnow()
                new_rates = mt5.copy_rates_range(symbol, timeframe, date_from_update, date_to_update)
                if new_rates is not None and len(new_rates) > 0:
                    df_new = pd.DataFrame(new_rates)
                    df_new['time'] = pd.to_datetime(df_new['time'], unit='s')
                    data_cache_instance.update_ohlc_cache(symbol, timeframe_str, df_new, max_bars_to_store)
                    cached_df, _ = data_cache_instance.get_cached_ohlc(symbol, timeframe_str)
                    logger.info(f"Fetched {len(df_new)} new bars for {symbol} on {timeframe_str} and updated cache")
                else:
                    logger.debug(f"No new bars for {symbol} on {timeframe_str} to update cache")

            if cached_df is not None and not cached_df.empty:
                required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
                if not all(col in cached_df.columns for col in required_columns):
                    logger.error(f"Missing columns in cached data for {symbol}: {cached_df.columns}")
                    return None
                logger.info(f"Returning {min(len(cached_df), bars)} cached bars for {symbol} on {timeframe_str}")
                return cached_df.tail(bars)[required_columns].copy()
            else:
                logger.error(f"No cached data available for {symbol} on {timeframe_str}")
                return None

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
        return None

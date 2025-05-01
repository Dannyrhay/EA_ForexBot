import MetaTrader5 as mt5
import pandas as pd
from utils.logging import setup_logging

logger = setup_logging()

def connect_mt5():
    """
    Initialize connection to MetaTrader 5 terminal.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        if not mt5.initialize(login=242854137, password="Danny@0011", server="Exness-MT5Trial"):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        logger.info("MT5 connection established")
        return True
    except Exception as e:
        logger.error(f"Error connecting to MT5: {e}")
        return False

def get_data(symbol, timeframe, bars, start_time=None):
    """
    Fetch historical data for a given symbol and timeframe.
    
    Args:
        symbol (str): Trading symbol (e.g., 'XAUUSDm')
        timeframe (int): Timeframe (e.g., mt5.TIMEFRAME_M5)
        bars (int): Number of bars to fetch
        start_time (datetime, optional): Start time for data fetching
    
    Returns:
        pd.DataFrame: Historical data with 'time' as a column or None if failed
    """
    try:
        if not mt5.symbol_info(symbol):
            logger.error(f"Symbol {symbol} not found in Market Watch")
            return None
        
        if start_time:
            rates = mt5.copy_rates_from(symbol, timeframe, start_time, bars)
        else:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        
        if rates is None or len(rates) == 0:
            logger.error(f"No data fetched for {symbol} on timeframe {timeframe}: {mt5.last_error()}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns in data for {symbol}: {df.columns}")
            return None
        
        logger.info(f"Successfully fetched {len(df)} bars for {symbol} on timeframe {timeframe}")
        return df[required_columns]
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None
import MetaTrader5 as mt5
import pandas as pd
from utils.logging import setup_logging

logger = setup_logging()

def connect_mt5(login=242854137, password="Danny@0011", server="Exness-MT5Trial"):
    """Connect to Exness MT5 server."""
    try:
        success = mt5.initialize(login=login, password=password, server=server)
        if not success:
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to retrieve account info")
            return False
        logger.info(f"Exness MT5 connection established: Account #{account_info.login}, Server: {server}")
        return True
    except Exception as e:
        logger.error(f"MT5 connection error: {e}")
        return False

def get_data(symbol, timeframe, bars):
    """Fetch historical data from Exness MT5."""
    try:
        if not mt5.symbol_info(symbol):
            logger.error(f"Symbol {symbol} not found in Market Watch")
            return None
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            logger.warning(f"No data fetched for {symbol} on timeframe {timeframe}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        logger.info(f"Successfully fetched {len(df)} bars for {symbol} on timeframe {timeframe}")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol} on timeframe {timeframe}: {e}")
        return None
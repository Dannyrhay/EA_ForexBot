import pandas as pd
from utils.logging import setup_logging
import os

logger = setup_logging()
trades = []
TRADE_LOG_FILE = "trades.csv"

def load_trades():
    """Load existing trades from CSV into the global trades list."""
    global trades
    try:
        if os.path.exists(TRADE_LOG_FILE):
            df = pd.read_csv(TRADE_LOG_FILE)
            trades = df.to_dict('records')
            logger.debug(f"Loaded {len(trades)} trades from {TRADE_LOG_FILE}")
        else:
            trades = []
            logger.debug(f"No existing trades file found at {TRADE_LOG_FILE}")
    except Exception as e:
        logger.error(f"Error loading trades from {TRADE_LOG_FILE}: {e}")
        trades = []

def save_trade(trade_data, update=False):
    """Save or update a trade in memory and append to CSV."""
    global trades
    try:
        # Load existing trades if not already loaded
        if not trades and os.path.exists(TRADE_LOG_FILE):
            load_trades()
        
        if update:
            for i, trade in enumerate(trades):
                if trade["order_id"] == trade_data["order_id"]:
                    trades[i].update(trade_data)
                    break
        else:
            # Check if trade already exists to avoid duplicates
            if not any(trade["order_id"] == trade_data["order_id"] for trade in trades):
                trades.append(trade_data)
        
        # Persist to CSV by reading existing file and appending
        try:
            if os.path.exists(TRADE_LOG_FILE):
                df_existing = pd.read_csv(TRADE_LOG_FILE)
                df_new = pd.DataFrame(trades)
                # Merge, keeping the latest data for each order_id
                df_combined = pd.concat([df_existing, df_new]).drop_duplicates(subset=['order_id'], keep='last')
            else:
                df_combined = pd.DataFrame(trades)
            
            # Ensure consistent data types
            for col in df_combined.columns:
                df_combined[col] = df_combined[col].astype(str) if df_combined[col].dtype == 'object' else df_combined[col]
            
            df_combined.to_csv(TRADE_LOG_FILE, index=False)
            logger.debug(f"Trade saved: {trade_data['status']} for {trade_data['symbol']}, Total trades: {len(trades)}")
        except Exception as e:
            logger.error(f"Error persisting trades to {TRADE_LOG_FILE}: {e}")
    except Exception as e:
        logger.error(f"Error saving trade: {e}")

def update_trade_status(order_id, update_data):
    """Update the status of a trade in memory and CSV."""
    global trades
    try:
        # Load existing trades if not already loaded
        if not trades and os.path.exists(TRADE_LOG_FILE):
            load_trades()
        
        # Update in-memory trades
        updated = False
        for i, trade in enumerate(trades):
            if trade["order_id"] == order_id:
                trades[i].update(update_data)
                updated = True
                break
        
        # Persist to CSV
        try:
            if os.path.exists(TRADE_LOG_FILE):
                df_existing = pd.read_csv(TRADE_LOG_FILE)
                df_new = pd.DataFrame(trades)
                # Merge, keeping the latest data for each order_id
                df_combined = pd.concat([df_existing, df_new]).drop_duplicates(subset=['order_id'], keep='last')
            else:
                df_combined = pd.DataFrame(trades)
            
            # Ensure consistent data types
            for col in df_combined.columns:
                df_combined[col] = df_combined[col].astype(str) if df_combined[col].dtype == 'object' else df_combined[col]
            
            df_combined.to_csv(TRADE_LOG_FILE, index=False)
            if updated:
                logger.debug(f"Trade updated: Order ID {order_id}, Status: {update_data['status']}, Total trades: {len(trades)}")
            else:
                logger.warning(f"No trade found to update for Order ID {order_id}")
        except Exception as e:
            logger.error(f"Error persisting trade update to {TRADE_LOG_FILE}: {e}")
    except Exception as e:
        logger.error(f"Error updating trade status for Order ID {order_id}: {e}")

def get_strategy_weights():
    """Calculate strategy weights based on historical performance."""
    try:
        # Load trades from CSV to ensure all historical data is considered
        if os.path.exists(TRADE_LOG_FILE):
            df = pd.read_csv(TRADE_LOG_FILE)
            df = df[df["status"] == "closed"]
        else:
            df = pd.DataFrame()
        
        if df.empty:
            default_weights = {
                "SupplyDemand": 1.0, "SMA": 1.0, "SMC": 1.0, "LiquiditySweep": 1.0,
                "FVG": 1.0, "Fibonacci": 1.0, "Candlestick": 1.0, "MalaysianSnR": 1.0,
                "BollingerBands": 1.0
            }
            logger.debug("No closed trades found, returning default strategy weights")
            return default_weights
        
        weights = {}
        for strategy in ["SupplyDemand", "SMA", "SMC", "LiquiditySweep", "FVG", 
                        "Fibonacci", "Candlestick", "MalaysianSnR", "BollingerBands"]:
            # Convert strategies column to list if stored as string
            df['strategies'] = df['strategies'].apply(lambda x: eval(x) if isinstance(x, str) else x)
            strategy_trades = df[df['strategies'].apply(lambda x: strategy in x)]
            if not strategy_trades.empty:
                win_rate = len(strategy_trades[strategy_trades['profit_loss'] > 0]) / len(strategy_trades)
                avg_profit = strategy_trades['profit_loss'].mean()
                weight = max(0.5, min(2.0, win_rate * (1 + avg_profit / 100)))
                weights[strategy] = weight
            else:
                weights[strategy] = 1.0
        logger.debug(f"Strategy weights: {weights}")
        return weights
    except Exception as e:
        logger.error(f"Error calculating strategy weights: {e}")
        default_weights = {
            "SupplyDemand": 1.0, "SMA": 1.0, "SMC": 1.0, "LiquiditySweep": 1.0,
            "FVG": 1.0, "Fibonacci": 1.0, "Candlestick": 1.0, "MalaysianSnR": 1.0,
            "BollingerBands": 1.0
        }
        return default_weights

# Load trades on module import
load_trades()
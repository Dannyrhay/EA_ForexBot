import pandas as pd
from utils.logging import setup_logging
import os
import ast # Using ast.literal_eval is safer than eval

logger = setup_logging()
trades = []  # In-memory cache of trades
TRADE_LOG_FILE = "trades.csv"

def load_trades():
    """Load existing trades from CSV into the global trades list."""
    global trades
    try:
        if os.path.exists(TRADE_LOG_FILE):
            df = pd.read_csv(TRADE_LOG_FILE)
            # Convert relevant columns to appropriate types
            if 'order_id' in df.columns:
                df['order_id'] = pd.to_numeric(df['order_id'], errors='coerce').fillna(0).astype(int)
            if 'profit_loss' in df.columns:
                df['profit_loss'] = pd.to_numeric(df['profit_loss'], errors='coerce').fillna(0.0)
            if 'account_balance' in df.columns:
                df['account_balance'] = pd.to_numeric(df['account_balance'], errors='coerce').fillna(0.0)
            if 'strategies' in df.columns:
                # Ensure strategies are read as strings, handling potential float/NaN from CSV
                df['strategies'] = df['strategies'].astype(str)


            trades = df.to_dict('records')
            logger.debug(f"Loaded {len(trades)} trades from {TRADE_LOG_FILE}")
        else:
            trades = []
            logger.debug(f"No existing trades file found at {TRADE_LOG_FILE}")
    except pd.errors.EmptyDataError:
        logger.info(f"{TRADE_LOG_FILE} is empty. Initializing empty trades list.")
        trades = []
    except Exception as e:
        logger.error(f"Error loading trades from {TRADE_LOG_FILE}: {e}", exc_info=True)
        trades = []

def save_trade(trade_data, update=False):
    """Save or update a trade in memory and append/update in CSV."""
    global trades
    try:
        if 'order_id' not in trade_data or trade_data['order_id'] is None:
            logger.error(f"Trade data missing order_id: {trade_data}")
            return
        trade_data['order_id'] = int(trade_data['order_id'])

        if not trades and os.path.exists(TRADE_LOG_FILE): 
            load_trades()
        
        found_for_update = False
        if update:
            for i, trade in enumerate(trades):
                if trade.get("order_id") == trade_data["order_id"]:
                    trades[i].update(trade_data)
                    found_for_update = True
                    break
            if not found_for_update: 
                trades.append(trade_data)
                logger.debug(f"Trade ID {trade_data['order_id']} not found for update, appended as new.")
        else: 
            existing_trade_index = -1
            for i, trade in enumerate(trades):
                if trade.get("order_id") == trade_data["order_id"]:
                    existing_trade_index = i
                    break
            if existing_trade_index != -1:
                trades[existing_trade_index].update(trade_data)
                logger.debug(f"Order ID {trade_data['order_id']} existed, record updated during append.")
            else:
                trades.append(trade_data)

        if trades:
            df_to_save = pd.DataFrame(trades)
            expected_cols = [
                "order_id", "symbol", "timeframe", "signal", "entry_price", "sl_price", "tp_price",
                "lot_size", "strategies", "entry_time", "status", "profit_loss", "account_balance",
                "failure_reason", "exit_time", "exit_reason" 
            ]
            for col in expected_cols:
                if col not in df_to_save.columns:
                    df_to_save[col] = pd.NA 
            df_to_save = df_to_save[expected_cols]
            if 'strategies' in df_to_save.columns:
                 # Ensure strategies are stored as strings (pandas default for lists might be non-standard)
                 df_to_save['strategies'] = df_to_save['strategies'].apply(
                     lambda x: str(x) if isinstance(x, list) else (str(x) if pd.notna(x) else '[]')
                 )


            df_to_save.to_csv(TRADE_LOG_FILE, index=False)
            logger.debug(f"Trades saved to {TRADE_LOG_FILE}. Total trades in memory: {len(trades)}")
        elif os.path.exists(TRADE_LOG_FILE):
            # If trades list is empty, make sure the CSV file is also empty
            with open(TRADE_LOG_FILE, 'w') as f:
                # Write header if you want to keep an empty file with headers
                # f.write(','.join(expected_cols) + '\n') 
                pass # Or just leave it empty
            logger.debug(f"{TRADE_LOG_FILE} cleared or ensured empty as in-memory trades list is empty.")

    except Exception as e:
        logger.error(f"Error saving trade (Order ID: {trade_data.get('order_id', 'N/A')}): {e}", exc_info=True)

def update_trade_status(order_id_to_update, update_data_dict):
    """Update the status and other details of a trade in memory and CSV."""
    try:
        order_id_to_update = int(order_id_to_update)
        trade_full_data = {"order_id": order_id_to_update}
        trade_full_data.update(update_data_dict)
        save_trade(trade_full_data, update=True) 
        logger.debug(f"Trade update processed for Order ID {order_id_to_update} with data: {update_data_dict}")
    except Exception as e:
        logger.error(f"Error in update_trade_status for Order ID {order_id_to_update}: {e}", exc_info=True)

def safe_parse_strategies(strat_data):
    """Safely parse the strategies entry, returning a list or empty list."""
    if isinstance(strat_data, list):
        return strat_data
    # Handle various forms of "empty" or NaN explicitly before attempting ast.literal_eval
    if pd.isna(strat_data) or strat_data == 'nan' or strat_data == '' or strat_data == '[]':
        return []
    if not isinstance(strat_data, str):
        logger.debug(f"Strategies data is not a string type for parsing (type: {type(strat_data)}, value: '{strat_data}'). Returning empty list.")
        return []
    try:
        # ast.literal_eval is safer for evaluating string representations of Python literals
        parsed = ast.literal_eval(strat_data)
        if isinstance(parsed, list):
            return parsed
        else:
            # If it parsed but not to a list (e.g. string was just "StrategyName")
            logger.warning(f"Parsed strategies string '{strat_data}' was not a list: {type(parsed)}. Returning empty list.")
            return []
    except (ValueError, SyntaxError, TypeError) as e:
        # If parsing fails (e.g., malformed string, or not a literal)
        logger.warning(f"Could not parse strategies string: '{strat_data}' (Error: {e}). Returning empty list.")
        return []


def get_strategy_weights():
    default_initial_weights = {
        "SupplyDemand": 1.0, "SMA": 1.0, "SMC": 1.0, "LiquiditySweep": 1.0,
        "FVG": 1.0, "Fibonacci": 1.0, "Candlestick": 1.0, "MalaysianSnR": 1.0,
        "BollingerBands": 1.0
    }
    try:
        trade_file_path = TRADE_LOG_FILE
        if not os.path.exists(trade_file_path):
            logger.info(f"{trade_file_path} not found. Returning default initial weights (all 1.0).")
            return default_initial_weights
        
        try:
            df = pd.read_csv(trade_file_path)
        except pd.errors.EmptyDataError:
            logger.info(f"{trade_file_path} is empty. Returning default initial weights (all 1.0).")
            return default_initial_weights


        if df.empty:
            logger.info("trades.csv is empty (after read). Returning default initial weights (all 1.0).")
            return default_initial_weights

        if 'profit_loss' in df.columns:
            df['profit_loss'] = pd.to_numeric(df['profit_loss'], errors='coerce').fillna(0.0)
        else:
            logger.warning("'profit_loss' column missing. Using default weights (1.0).")
            return default_initial_weights

        if 'status' in df.columns:
            # *** FIX: Use .copy() here to avoid SettingWithCopyWarning later ***
            df_closed = df[df["status"].isin(["closed", "closed_auto"])].copy()
        else:
            logger.warning("'status' column missing. Using default weights (1.0).")
            return default_initial_weights

        if df_closed.empty:
            logger.info("No closed trades found in history. Returning default initial weights (all 1.0).")
            return default_initial_weights
        
        if 'strategies' not in df_closed.columns:
            logger.warning("'strategies' column missing. Using default weights (1.0).")
            return default_initial_weights

        # Ensure the 'strategies' column is string type before applying safe_parse_strategies
        df_closed.loc[:, 'strategies'] = df_closed['strategies'].astype(str)
        df_closed.loc[:, 'parsed_strategies'] = df_closed['strategies'].apply(safe_parse_strategies)
        
        weights = {}
        all_strategy_names = list(default_initial_weights.keys())

        for strategy_name in all_strategy_names:
            # Filter trades where the current strategy_name is in the 'parsed_strategies' list
            strategy_trades = df_closed[df_closed['parsed_strategies'].apply(lambda strat_list: isinstance(strat_list, list) and strategy_name in strat_list)]
            
            if not strategy_trades.empty:
                win_count = len(strategy_trades[strategy_trades['profit_loss'] > 0])
                total_trades_for_strat = len(strategy_trades)
                
                if total_trades_for_strat == 0:
                    weights[strategy_name] = 1.0 
                    continue

                win_rate = win_count / total_trades_for_strat
                avg_profit = strategy_trades['profit_loss'].mean() 
                
                profit_factor = 1.0
                if not pd.isna(avg_profit) and avg_profit != 0:
                     profit_influence = (avg_profit / 100.0) * 0.2 # Example: scale avg profit by 0.002
                     profit_factor = 1.0 + max(min(profit_influence, 0.2), -0.2) # Bound influence to +/- 20%

                calculated_weight = win_rate * profit_factor
                weights[strategy_name] = max(0.5, min(2.0, calculated_weight)) # Clamp final weight
            else:
                # If a strategy has no historical trades, assign it the default initial weight of 1.0
                weights[strategy_name] = 1.0 
        
        logger.debug(f"Calculated strategy weights based on history: {weights}")
        return weights
    except Exception as e:
        logger.error(f"Error calculating strategy weights: {e}", exc_info=True)
        return default_initial_weights

load_trades()

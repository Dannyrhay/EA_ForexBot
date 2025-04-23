import pandas as pd
from utils.logging import setup_logging

logger = setup_logging()
trades = []

def save_trade(trade_data, update=False):
    """Save or update a trade in memory."""
    try:
        global trades
        if update:
            for i, trade in enumerate(trades):
                if trade["order_id"] == trade_data["order_id"]:
                    trades[i].update({
                        "status": trade_data["status"],
                        "profit_loss": trade_data["profit_loss"],
                        "account_balance": trade_data["account_balance"],
                        "exit_time": trade_data["exit_time"]
                    })
                    break
        else:
            trades.append(trade_data)
        logger.debug(f"Trade saved: {trade_data['status']} for {trade_data['symbol']}")
    except Exception as e:
        logger.error(f"Error saving trade: {e}")

def get_strategy_weights():
    """Calculate strategy weights based on historical performance."""
    try:
        if not trades:
            return {"SupplyDemand": 1.0, "SMA": 1.0, "SMC": 1.0, "LiquiditySweep": 1.0, "FVG": 1.0, "Fibonacci": 1.0}

        df = pd.DataFrame([t for t in trades if t["status"] == "closed"])
        if df.empty:
            return {"SupplyDemand": 1.0, "SMA": 1.0, "SMC": 1.0, "LiquiditySweep": 1.0, "FVG": 1.0, "Fibonacci": 1.0}

        weights = {}
        for strategy in ["SupplyDemand", "SMA", "SMC", "LiquiditySweep", "FVG", "Fibonacci"]:
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
        return {"SupplyDemand": 1.0, "SMA": 1.0, "SMC": 1.0, "LiquiditySweep": 1.0, "FVG": 1.0, "Fibonacci": 1.0}

import os
import logging
from utils.db_connector import MongoDBConnection
from datetime import datetime, timedelta, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_trades():
    db_conn = MongoDBConnection.connect()
    if db_conn is None:
        logger.error("Failed to connect to MongoDB.")
        return

    trades_collection = MongoDBConnection.get_trades_collection()

    # 1. Check total count
    total_count = trades_collection.count_documents({})
    logger.info(f"Total documents in trades collection: {total_count}")

    # 2. Check for any trades (limit 5)
    any_trades = list(trades_collection.find({}).limit(5))
    logger.info(f"Found {len(any_trades)} sample trades (any status):")
    for t in any_trades:
        logger.info(f"ID: {t.get('order_id')}, Status: {t.get('status')}, Exit Time: {t.get('exit_time')}, P/L: {t.get('profit_loss')}")

    # 3. Check specifically for the query that is failing
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
    query = {
        "status": {"$in": ["closed", "closed_auto"]},
        "profit_loss": {"$exists": True, "$type": "number"},
        "strategies": {"$exists": True},
        "exit_time": {"$gte": cutoff_date}
    }
    matching_count = trades_collection.count_documents(query)
    logger.info(f"Documents matching the STRICT query: {matching_count}")

    if matching_count == 0:
        logger.info("Debugging why strict query fails...")
        # Check individual conditions
        c1 = trades_collection.count_documents({"status": {"$in": ["closed", "closed_auto"]}})
        logger.info(f"  Matches Status ['closed', 'closed_auto']: {c1}")

        c2 = trades_collection.count_documents({"profit_loss": {"$exists": True, "$type": "number"}})
        logger.info(f"  Matches Profit/Loss is Number: {c2}")

        c3 = trades_collection.count_documents({"strategies": {"$exists": True}})
        logger.info(f"  Matches Strategies exists: {c3}")

        c4 = trades_collection.count_documents({"exit_time": {"$gte": cutoff_date}})
        logger.info(f"  Matches Exit Time >= {cutoff_date}: {c4}")

if __name__ == "__main__":
    inspect_trades()

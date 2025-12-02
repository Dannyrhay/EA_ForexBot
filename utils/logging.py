import logging
from datetime import datetime
import pytz

class UTCTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        utc_dt = datetime.fromtimestamp(record.created, tz=pytz.UTC)
        if datefmt:
            return utc_dt.strftime(datefmt)
        return utc_dt.strftime('%Y-%m-%d %H:%M:%S %Z')

def setup_logging():
    """Set up logging to file and console with UTC timestamps."""
    # Configure root logger to capture logs from all modules
    logger = logging.getLogger()
    if logger.handlers:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    formatter = UTCTimeFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %Z'
    )

    # File Handler
    file_handler = logging.FileHandler('logs/trading_ea.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    return logger
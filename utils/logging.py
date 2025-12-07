import logging
from datetime import datetime
import pytz
from logging.handlers import RotatingFileHandler

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
    
    # Set root logger to INFO to reduce verbosity
    logger.setLevel(logging.DEBUG)

    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('pymongo').setLevel(logging.WARNING)

    formatter = UTCTimeFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %Z'
    )

    # File Handler with Rotation
    file_handler = RotatingFileHandler('logs/trading_ea.log', maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
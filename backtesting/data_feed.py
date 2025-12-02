import pandas as pd
import numpy as np
from datetime import datetime

class DataFeed:
    """
    Feeds historical data to the backtester as a stream of events.
    Simulates "ticks" from M1 bars to allow for intra-bar execution.
    """
    def __init__(self, data, symbol):
        """
        Args:
            data (pd.DataFrame): DataFrame with 'time', 'open', 'high', 'low', 'close'.
            symbol (str): Symbol name (e.g., 'EURUSD').
        """
        self.data = data.reset_index(drop=True)
        self.symbol = symbol
        self.current_idx = 0

    def get_next_event(self):
        """
        Yields the next market event.
        To simulate realism, we break an M1 candle into 4 "pseudo-ticks":
        1. Open
        2. Low (if Bearish) or High (if Bullish) - simplified path
        3. High (if Bearish) or Low (if Bullish)
        4. Close

        For now, we'll use a simpler OHLC sequence: Open -> Low -> High -> Close
        This ensures we hit the wicks.
        """
        if self.current_idx >= len(self.data):
            return None

        row = self.data.iloc[self.current_idx]

        # Create 4 pseudo-ticks for this bar
        # Event format: (timestamp, price, type)
        # Type: 'tick'

        timestamp = row['time']

        # 1. Open
        yield {'type': 'tick', 'symbol': self.symbol, 'time': timestamp, 'price': row['open'], 'kind': 'open'}

        # 2. Low
        yield {'type': 'tick', 'symbol': self.symbol, 'time': timestamp, 'price': row['low'], 'kind': 'low'}

        # 3. High
        yield {'type': 'tick', 'symbol': self.symbol, 'time': timestamp, 'price': row['high'], 'kind': 'high'}

        # 4. Close (New Bar Event effectively)
        yield {'type': 'tick', 'symbol': self.symbol, 'time': timestamp, 'price': row['close'], 'kind': 'close'}

        # 5. End of Bar Event (Trigger for Strategy Logic)
        yield {'type': 'bar_close', 'symbol': self.symbol, 'time': timestamp, 'data': row}

        self.current_idx += 1

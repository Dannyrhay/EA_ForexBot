from .base_strategy import BaseStrategy
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class SMAStrategy(BaseStrategy):
    def __init__(self, name, short_period, long_period):
        super().__init__(name)
        self.short_period = short_period
        self.long_period = long_period

    def get_signal(self, data, symbol=None, timeframe=None):
        """
        Generate trading signal based on SMA crossover and trend continuation.
        - Entry signal on the exact bar of the crossover.
        - A lower-strength 'hold' signal that confirms the trend direction
          as long as the short SMA remains above/below the long SMA.
        """
        try:
            df = pd.DataFrame(data)
            if len(df) < self.long_period + 2:
                return ('hold', 0.0)

            df['sma_short'] = df['close'].rolling(window=self.short_period).mean()
            df['sma_long'] = df['close'].rolling(window=self.long_period).mean()

            # Check for NaN values in the latest two periods
            if df['sma_short'].iloc[-2:].isnull().any() or df['sma_long'].iloc[-2:].isnull().any():
                return ('hold', 0.0)

            # --- Entry Signals (Crossover) ---
            # Bullish crossover
            if (df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1] and
                    df['sma_short'].iloc[-2] <= df['sma_long'].iloc[-2]):
                logger.info(f"SMAStrategy ({symbol} {timeframe}): BUY signal on crossover.")
                return ('buy', 0.8) # High confidence for entry signal

            # Bearish crossover
            elif (df['sma_short'].iloc[-1] < df['sma_long'].iloc[-1] and
                  df['sma_short'].iloc[-2] >= df['sma_long'].iloc[-2]):
                logger.info(f"SMAStrategy ({symbol} {timeframe}): SELL signal on crossover.")
                return ('sell', 0.8) # High confidence for entry signal

            # --- Trend Continuation Confirmation (as per thesis diagram) ---
            # This logic provides context about the current prevailing trend according to the SMAs.
            # While it returns 'hold', a more advanced consensus system could use this
            # to validate trades from other strategies or to hold existing positions.
            elif df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1]:
                # The trend is currently bullish, but it's not a new crossover event.
                # Returning 'hold' but logging the state for clarity.
                logger.debug(f"SMAStrategy ({symbol} {timeframe}): Hold signal. Trend continuation is BULLISH.")
                # We could return a tuple like ('hold_buy_trend', 0.2) for a more complex system.
                return ('hold', 0.0)
            elif df['sma_short'].iloc[-1] < df['sma_long'].iloc[-1]:
                # The trend is currently bearish.
                logger.debug(f"SMAStrategy ({symbol} {timeframe}): Hold signal. Trend continuation is BEARISH.")
                return ('hold', 0.0)

            return ('hold', 0.0)

        except Exception as e:
            logger.error(f"SMAStrategy ({symbol} {timeframe}): Error in get_signal: {e}", exc_info=True)
            return ('hold', 0.0)

    def set_config(self, config: dict):
        """Sets strategy parameters from the main configuration dictionary."""
        self.short_period = config.get('sma_short_period', self.short_period)
        self.long_period = config.get('sma_long_period', self.long_period)
        logger.debug(f"SMAStrategy config set: Short={self.short_period}, Long={self.long_period}")


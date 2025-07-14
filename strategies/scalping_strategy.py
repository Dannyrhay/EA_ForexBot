import pandas as pd
import logging
from .base_strategy import BaseStrategy
import numpy as np

logger = logging.getLogger(__name__)

class ScalpingStrategy(BaseStrategy):
    """
    Implements the 1-Minute Scalping Strategy.
    This is a trend-following strategy that uses a dual EMA system for trend direction
    and the Stochastic Oscillator to time entries during pullbacks.
    """
    def __init__(self, name, **params):
        super().__init__(name)
        # Default parameters from the PDF
        self.ema_short_period = 50
        self.ema_long_period = 100
        self.stoch_k_period = 5
        self.stoch_d_period = 3
        self.stoch_slowing = 3
        self.stoch_oversold = 20
        self.stoch_overbought = 80
        self.signal_strength = 0.9

        # Apply any overrides from the constructor
        if params:
            self.set_config(params)

    def set_config(self, config: dict):
        """Load parameters from a dictionary, typically from config.json."""
        params = config.get('scalping_strategy_params', config)
        self.ema_short_period = params.get('ema_short_period', self.ema_short_period)
        self.ema_long_period = params.get('ema_long_period', self.ema_long_period)
        self.stoch_k_period = params.get('stoch_k_period', self.stoch_k_period)
        self.stoch_d_period = params.get('stoch_d_period', self.stoch_d_period)
        self.stoch_slowing = params.get('stoch_slowing', self.stoch_slowing)
        self.stoch_oversold = params.get('stoch_oversold', self.stoch_oversold)
        self.stoch_overbought = params.get('stoch_overbought', self.stoch_overbought)
        self.signal_strength = params.get('signal_strength', self.signal_strength)
        logger.debug("ScalpingStrategy config loaded.")

    def _calculate_stochastic(self, data: pd.DataFrame):
        """Helper to calculate Stochastic Oscillator."""
        low_min = data['low'].rolling(window=self.stoch_k_period).min()
        high_max = data['high'].rolling(window=self.stoch_k_period).max()
        data['stoch_k'] = 100 * (data['close'] - low_min) / (high_max - low_min)
        # The PDF uses (5,3,3) which typically means %K is smoothed by 3 periods before %D is calculated.
        data['stoch_k_smoothed'] = data['stoch_k'].rolling(window=self.stoch_slowing).mean()
        data['stoch_d'] = data['stoch_k_smoothed'].rolling(window=self.stoch_d_period).mean()
        return data

    def get_signal(self, data: pd.DataFrame, symbol: str = None, timeframe: str = None):
        """Generates a trade signal based on the trend-following rules."""
        required_len = max(self.ema_long_period, self.stoch_k_period + self.stoch_slowing + self.stoch_d_period)
        if len(data) < required_len:
            return ('hold', 0.0)

        df = data.copy()
        df['ema_short'] = df['close'].ewm(span=self.ema_short_period, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.ema_long_period, adjust=False).mean()
        df = self._calculate_stochastic(df)

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        if pd.isna(latest.ema_short) or pd.isna(latest.stoch_k_smoothed):
            return ('hold', 0.0)

        # --- Buy Signal Logic ---
        # 1. Trend is up
        is_uptrend = latest.ema_short > latest.ema_long
        # 2. Price returns to the EMAs (a simple check: low of the candle touches the short EMA)
        price_pullback_buy = latest.low <= latest.ema_short
        # 3. Stochastic crosses above oversold level
        stoch_confirmation_buy = latest.stoch_k_smoothed > self.stoch_oversold and previous.stoch_k_smoothed <= self.stoch_oversold

        if is_uptrend and price_pullback_buy and stoch_confirmation_buy:
            logger.info(f"ScalpingStrategy ({symbol} {timeframe}): BUY signal. Uptrend, pullback, and Stoch confirmation.")
            return ('buy', self.signal_strength)

        # --- Sell Signal Logic ---
        # 1. Trend is down
        is_downtrend = latest.ema_short < latest.ema_long
        # 2. Price returns to the EMAs
        price_pullback_sell = latest.high >= latest.ema_short
        # 3. Stochastic crosses below overbought level
        stoch_confirmation_sell = latest.stoch_k_smoothed < self.stoch_overbought and previous.stoch_k_smoothed >= self.stoch_overbought

        if is_downtrend and price_pullback_sell and stoch_confirmation_sell:
            logger.info(f"ScalpingStrategy ({symbol} {timeframe}): SELL signal. Downtrend, pullback, and Stoch confirmation.")
            return ('sell', self.signal_strength)

        return ('hold', 0.0)

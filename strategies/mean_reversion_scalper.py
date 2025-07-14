# strategies/mean_reversion_scalper.py

import pandas as pd
import logging
from .base_strategy import BaseStrategy
import numpy as np

logger = logging.getLogger(__name__)

class MeanReversionScalper(BaseStrategy):
   
    def __init__(self, name, **params):
        super().__init__(name)
        # Default parameters from the PDF
        self.bb_window = 20
        self.bb_std_dev = 2.0
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.stoch_k_period = 5
        self.stoch_d_period = 3
        self.stoch_slowing = 3
        self.stoch_overbought = 80
        self.stoch_oversold = 20
        self.signal_strength = 0.85

        # Apply any overrides from the constructor
        if params:
            self.set_config(params)

    def set_config(self, config: dict):
        """Load parameters from a dictionary, typically from config.json."""
        params = config.get('mean_reversion_scalper_params', config)
        self.bb_window = params.get('bb_window', self.bb_window)
        self.bb_std_dev = params.get('bb_std_dev', self.bb_std_dev)
        self.rsi_period = params.get('rsi_period', self.rsi_period)
        self.rsi_overbought = params.get('rsi_overbought', self.rsi_overbought)
        self.rsi_oversold = params.get('rsi_oversold', self.rsi_oversold)
        self.stoch_k_period = params.get('stoch_k_period', self.stoch_k_period)
        self.stoch_d_period = params.get('stoch_d_period', self.stoch_d_period)
        self.stoch_slowing = params.get('stoch_slowing', self.stoch_slowing)
        self.stoch_overbought = params.get('stoch_overbought', self.stoch_overbought)
        self.stoch_oversold = params.get('stoch_oversold', self.stoch_oversold)
        self.signal_strength = params.get('signal_strength', self.signal_strength)
        logger.debug("MeanReversionScalper config loaded.")

    def _calculate_indicators(self, data: pd.DataFrame):
        """Helper to calculate all necessary indicators."""
        # Bollinger Bands
        data['bb_ma'] = data['close'].rolling(window=self.bb_window).mean()
        data['bb_std'] = data['close'].rolling(window=self.bb_window).std()
        data['bb_upper'] = data['bb_ma'] + (data['bb_std'] * self.bb_std_dev)
        data['bb_lower'] = data['bb_ma'] - (data['bb_std'] * self.bb_std_dev)

        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=self.rsi_period - 1, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=self.rsi_period - 1, adjust=False).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        low_min = data['low'].rolling(window=self.stoch_k_period).min()
        high_max = data['high'].rolling(window=self.stoch_k_period).max()
        data['stoch_k'] = 100 * (data['close'] - low_min) / (high_max - low_min)
        data['stoch_d'] = data['stoch_k'].rolling(window=self.stoch_d_period).mean()

        return data

    def get_signal(self, data: pd.DataFrame, symbol: str = None, timeframe: str = None):
        """Generates a trade signal based on the mean-reversion rules."""
        required_len = max(self.bb_window, self.rsi_period, self.stoch_k_period) + self.stoch_d_period
        if len(data) < required_len:
            return ('hold', 0.0)

        df = self._calculate_indicators(data.copy())

        latest = df.iloc[-1]

        # Check for NaN values
        if pd.isna(latest.bb_upper) or pd.isna(latest.rsi) or pd.isna(latest.stoch_k):
            return ('hold', 0.0)

        # --- Buy Signal Logic ---
        price_below_bb = latest['close'] <= latest['bb_lower']
        rsi_oversold = latest['rsi'] < self.rsi_oversold
        stoch_oversold = latest['stoch_k'] < self.stoch_oversold

        if price_below_bb and rsi_oversold and stoch_oversold:
            logger.info(f"MeanReversionScalper ({symbol} {timeframe}): BUY signal. Price below BB, RSI & Stoch oversold.")
            return ('buy', self.signal_strength)

        # --- Sell Signal Logic ---
        price_above_bb = latest['close'] >= latest['bb_upper']
        rsi_overbought = latest['rsi'] > self.rsi_overbought
        stoch_overbought = latest['stoch_k'] > self.stoch_overbought

        if price_above_bb and rsi_overbought and stoch_overbought:
            logger.info(f"MeanReversionScalper ({symbol} {timeframe}): SELL signal. Price above BB, RSI & Stoch overbought.")
            return ('sell', self.signal_strength)

        return ('hold', 0.0)

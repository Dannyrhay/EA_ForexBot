import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RegimeFilter:
    """
    Classifies the market regime into:
    - TRENDING_BULL
    - TRENDING_BEAR
    - RANGING
    - HIGH_VOLATILITY (Unsafe)
    """
    def __init__(self, adx_period=14, adx_threshold=25, volatility_window=20):
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volatility_window = volatility_window

    def get_regime(self, df: pd.DataFrame):
        """
        Analyzes the dataframe and returns the current regime string.
        """
        if len(df) < 50:
            return "UNCERTAIN"

        # 1. Calculate ADX
        # Simplified ADX calculation for regime detection
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(self.adx_period).mean()

        plus_di = 100 * (plus_dm.ewm(alpha=1/self.adx_period).mean() / atr)
        minus_di = 100 * (abs(minus_dm).ewm(alpha=1/self.adx_period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(self.adx_period).mean().iloc[-1]

        # 2. Calculate Volatility (ATR / Price)
        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]
        volatility_pct = (current_atr / current_price) * 100

        # High Volatility Check (e.g., > 0.5% movement per bar on M5 is crazy)
        if volatility_pct > 0.5:
            return "HIGH_VOLATILITY"

        # 3. Determine Regime
        if adx > self.adx_threshold:
            if plus_di.iloc[-1] > minus_di.iloc[-1]:
                return "TRENDING_BULL"
            else:
                return "TRENDING_BEAR"
        else:
            return "RANGING"

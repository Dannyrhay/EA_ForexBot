# strategies/ml_prediction_strategy.py
from .base_strategy import BaseStrategy
import pandas as pd
from utils.feature_engineering import extract_ml_features

class MLPredictionStrategy(BaseStrategy):
    """
    A strategy that generates trading signals directly from the predictions
    of a machine learning model, as described in the thesis.
    """
    def __init__(self, name, bot_instance):
        """
        Initializes the MLPredictionStrategy.

        Args:
            name (str): The name of the strategy.
            bot_instance (TradingBot): The main bot instance to access ML validator,
                                       config, and helper methods.
        """
        super().__init__(name)
        self.bot = bot_instance
        self.ml_validator = bot_instance.ml_validator
        self.config = bot_instance.config

    def get_signal(self, data: pd.DataFrame, symbol: str = None, timeframe: str = None):
        """
        Generates a signal based on direct ML model prediction.

        Returns:
            tuple: ('buy'/'sell'/'hold', confidence_score)
        """
        if self.ml_validator is None:
            return ('hold', 0.0)

        # 1. Get Prediction Thresholds from config
        buy_threshold = self.config.get('ml_confidence_thresholds', {}).get('buy', {}).get(symbol, 0.7)
        sell_threshold = self.config.get('ml_confidence_thresholds', {}).get('sell', {}).get(symbol, 0.7)

        # 2. Check for a BUY signal
        if self.ml_validator.is_fitted(symbol, 'buy'):
            dxy_data = self.bot.get_dxy_data_for_correlation(data.copy()) if symbol.upper() == 'XAUUSDM' else None
            buy_features = extract_ml_features(symbol, data.copy(), 'buy', self.bot, dxy_data)

            if buy_features:
                try:
                    buy_prob = self.ml_validator.predict_proba(symbol, [buy_features], 'buy')[0][1]
                    if buy_prob >= buy_threshold:
                        # If buy probability is high, no need to check sell
                        return ('buy', buy_prob)
                except Exception as e:
                    self.bot.logger.error(f"MLPredictionStrategy: Error predicting 'buy' for {symbol}: {e}")

        # 3. Check for a SELL signal
        if self.ml_validator.is_fitted(symbol, 'sell'):
            dxy_data = self.bot.get_dxy_data_for_correlation(data.copy()) if symbol.upper() == 'XAUUSDM' else None
            sell_features = extract_ml_features(symbol, data.copy(), 'sell', self.bot, dxy_data)

            if sell_features:
                try:
                    sell_prob = self.ml_validator.predict_proba(symbol, [sell_features], 'sell')[0][1]
                    if sell_prob >= sell_threshold:
                        return ('sell', sell_prob)
                except Exception as e:
                    self.bot.logger.error(f"MLPredictionStrategy: Error predicting 'sell' for {symbol}: {e}")

        # 4. If no signal meets the threshold, hold
        return ('hold', 0.0)

    def set_config(self, config: dict):
        """Updates the internal configuration."""
        self.config = config

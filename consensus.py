import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from models import TradeSignal
from utils.feature_engineering import extract_ml_features
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

def get_strategy_weights(lookback_days=30):
    """
    Placeholder for fetching dynamic strategy weights.
    In a real implementation, this would query the database for recent performance.
    """
    # TODO: Implement actual DB query
    return {}

class ConsensusEngine:
    def __init__(self, strategies: List[BaseStrategy], config: dict, ml_validator=None, regime_filter=None):
        self.strategies = strategies
        self.config = config
        self.ml_validator = ml_validator
        self.regime_filter = regime_filter
        self.weights = get_strategy_weights(self.config.get('strategy_weighting_lookback_days', 30))

        # Cache for ADX strategy if present
        self.adx_strategy = next((s for s in strategies if s.name == "ADX"), None)

    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> TradeSignal:
        """
        Orchestrates the signal generation and consensus process.
        """
        if data is None or data.empty:
            return TradeSignal(symbol, timeframe, "Consensus", "HOLD", 0.0, 0.0)

        # 1. Detect Regime
        regime = "unknown"
        if self.regime_filter:
            regime = self.regime_filter.get_regime(data)
            logger.debug(f"Market Regime for {symbol}: {regime}")

        # 2. Pre-calculate Global Indicators (ADX)
        adx_context = self._get_adx_context(data)

        # 3. Collect Signals from Strategies
        collected_signals: List[TradeSignal] = []
        for strategy in self.strategies:
            try:
                # Skip ADX - it's only used as a filter, not a signal generator
                if strategy.name == "ADX":
                    continue

                # Skip strategies not suited for current regime (if logic exists)
                # For now, we run all and let weights handle it, or add explicit regime check here

                # Pass context to strategy if it accepts it
                # Note: Strategies currently return tuples/dicts. We need to adapt them.
                # Ideally, strategies should return TradeSignal objects.
                # For backward compatibility, we wrap the result.

                raw_result = strategy.get_signal(data, symbol=symbol, timeframe=timeframe)

                signal_obj = self._normalize_signal(raw_result, strategy.name, symbol, timeframe, data.iloc[-1]['close'])

                if signal_obj.signal_type != 'HOLD':
                    # Apply ADX Filter
                    if self._check_adx_filter(signal_obj, adx_context, symbol):
                        collected_signals.append(signal_obj)
                    else:
                        logger.debug(f"Signal {signal_obj.strategy_name} {signal_obj.signal_type} filtered by ADX.")

            except Exception as e:
                logger.error(f"Error getting signal from {strategy.name}: {e}", exc_info=True)

        # 4. Calculate Weighted Consensus
        consensus_signal = self._calculate_consensus(collected_signals, symbol, timeframe, data)

        return consensus_signal

    def _get_adx_context(self, data: pd.DataFrame) -> dict:
        context = {'value': 0, 'plus_di': 0, 'minus_di': 0, 'trend': 'neutral'}
        if self.adx_strategy:
            try:
                vals = self.adx_strategy.get_indicator_values(data.copy())
                if vals and 'adx' in vals and not vals['adx'].empty:
                    context['value'] = vals['adx'].iloc[-1]
                    context['plus_di'] = vals['plus_di'].iloc[-1]
                    context['minus_di'] = vals['minus_di'].iloc[-1]
                    context['trend'] = 'bullish' if context['plus_di'] > context['minus_di'] else 'bearish'
            except Exception as e:
                logger.error(f"Error calculating ADX context: {e}")
        return context

    def _normalize_signal(self, raw_result, strategy_name, symbol, timeframe, current_price) -> TradeSignal:
        """Converts legacy tuple/dict returns to TradeSignal object."""
        sig_type = 'HOLD'
        strength = 0.0
        params = {}

        if isinstance(raw_result, tuple) and len(raw_result) == 3:
            s, st, p = raw_result
            sig_type = s.upper()
            strength = st
            params = p if p else {}
        elif isinstance(raw_result, dict) and 'entry_price' in raw_result:
            # Handle limit order setups (like from SMC)
            import MetaTrader5 as mt5 # Local import to avoid circular dependency issues if any
            sig_type = 'BUY' if raw_result.get('type') == mt5.ORDER_TYPE_BUY_LIMIT else 'SELL'
            strength = raw_result.get('strength', 0.9)
            params = raw_result

        # Map 'buy'/'sell' to 'BUY'/'SELL'
        if sig_type.lower() == 'buy': sig_type = 'BUY'
        if sig_type.lower() == 'sell': sig_type = 'SELL'

        return TradeSignal(
            symbol=symbol,
            timeframe=timeframe,
            strategy_name=strategy_name,
            signal_type=sig_type,
            strength=strength,
            price=params.get('entry_price', current_price),
            sl=params.get('sl'),
            tp=params.get('tp'),
            metadata=params
        )

    def _check_adx_filter(self, signal: TradeSignal, adx_context: dict, symbol: str) -> bool:
        cfg = self.config.get('adx_signal_filter', {})
        if not cfg.get('enabled', False):
            return True

        if signal.strategy_name == "ADX": # Don't filter ADX strategy with itself
            return True

        min_adx = cfg.get(f"min_adx_for_entry_{symbol}", cfg.get('min_adx_for_entry', 20))

        if adx_context['value'] < min_adx:
            return False

        if cfg.get('require_di_confirmation', True):
            if signal.signal_type == 'BUY' and adx_context['plus_di'] <= adx_context['minus_di']:
                return False
            if signal.signal_type == 'SELL' and adx_context['minus_di'] <= adx_context['plus_di']:
                return False

        return True

    def _calculate_consensus(self, signals: List[TradeSignal], symbol: str, timeframe: str, data: pd.DataFrame) -> TradeSignal:
        buy_score = 0.0
        sell_score = 0.0
        best_buy_params = None
        best_sell_params = None

        # Calculate Volatility for Threshold
        volatility = 0.0
        if len(data) >= 20:
            volatility = (data['high'].iloc[-20:].max() - data['low'].iloc[-20:].min()) / data['close'].iloc[-1]

        threshold_cfg = self.config.get('consensus_threshold', {}).get(symbol, self.config.get('consensus_threshold', {}).get("default", {"low_vol": 1.0, "high_vol": 1.5, "vol_split": 0.005}))
        threshold = threshold_cfg["low_vol"] if volatility < threshold_cfg["vol_split"] else threshold_cfg["high_vol"]

        for sig in signals:
            # Dynamic Weight
            perf_weight = self.weights.get(sig.strategy_name, 1.0)

            # Static Boost
            boost_cfg = self.config.get('strategy_boost_factor', {}).get(sig.strategy_name, 1.0)
            boost = boost_cfg.get(timeframe, boost_cfg.get('default', 1.0)) if isinstance(boost_cfg, dict) else boost_cfg

            final_weight = perf_weight * boost
            contribution = sig.strength * final_weight

            if sig.signal_type == 'BUY':
                buy_score += contribution
                if not best_buy_params: best_buy_params = sig # Take first valid params
            elif sig.signal_type == 'SELL':
                sell_score += contribution
                if not best_sell_params: best_sell_params = sig

        # Determine Winner
        dominance = self.config.get('consensus_dominance_factor', 1.8)
        final_signal = 'HOLD'
        winning_sig_obj = None

        if buy_score >= threshold and buy_score > sell_score * dominance:
            final_signal = 'BUY'
            winning_sig_obj = best_buy_params
        elif sell_score >= threshold and sell_score > buy_score * dominance:
            final_signal = 'SELL'
            winning_sig_obj = best_sell_params

        if final_signal == 'HOLD' or not winning_sig_obj:
            if max(buy_score, sell_score) > 0:
                logger.info(f"Consensus HOLD for {symbol} ({timeframe}): Score {max(buy_score, sell_score):.2f} < Threshold {threshold:.2f} (Buy: {buy_score:.2f}, Sell: {sell_score:.2f})")
            return TradeSignal(symbol, timeframe, "Consensus", "HOLD", 0.0, data.iloc[-1]['close'])

        # ML Validation
        if self.ml_validator:
            if not self._validate_with_ml(symbol, final_signal, data):
                logger.info(f"ML Vetoed Consensus {final_signal}")
                return TradeSignal(symbol, timeframe, "Consensus", "HOLD", 0.0, data.iloc[-1]['close'])

        # Return Final Consensus Signal
        # We preserve the SL/TP/Price from the best contributing strategy
        return TradeSignal(
            symbol=symbol,
            timeframe=timeframe,
            strategy_name="Consensus",
            signal_type=final_signal,
            strength=max(buy_score, sell_score), # Aggregate strength
            price=winning_sig_obj.price,
            sl=winning_sig_obj.sl,
            tp=winning_sig_obj.tp,
            metadata={'buy_score': buy_score, 'sell_score': sell_score, 'volatility': volatility}
        )

    def _validate_with_ml(self, symbol: str, signal_type: str, data: pd.DataFrame) -> bool:
        try:
            if not self.ml_validator.is_fitted(symbol, signal_type.lower()):
                return True # Pass if no model

            # Extract features
            # Note: We need to ensure this matches exactly how the model was trained
            # Ideally, we pass the bot instance or a feature extractor to the engine
            # For now, we use the standalone function, assuming it has what it needs
            # or we might need to pass 'self' if it needs bot state.
            # The standalone 'extract_ml_features' in utils might need the bot instance.
            # This is a dependency injection point we need to handle.

            # For this implementation, we assume we can call it.
            # If it needs 'bot_instance', we might need to pass it in __init__.
            # Let's assume for now we can get by or we'll fix the import.

            # TODO: Fix dependency on 'bot_instance' for feature extraction if needed.
            # For now, we'll return True to not block, but this needs to be wired up.
            return True

        except Exception as e:
            logger.error(f"ML Validation Error: {e}")
            return True # Fail open? Or fail closed?

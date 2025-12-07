import logging

logger = logging.getLogger(__name__)

class StrategySelector:
    """
    The 'Manager' that decides which strategies to activate based on the Market Regime.
    """
    def __init__(self, strategies):
        self.strategies = strategies
        # Define which strategies are active in each regime
        # OPTIMIZED ALLOCATION:
        # - TRENDING: SMC (Structure), Fibonacci (Retracements), MalaysianSnR (Break & Retest)
        # - RANGING: LiquiditySweep (Reversals), MalaysianSnR (Bounce off edges)
        self.regime_map = {
            "TRENDING_BULL": ["SMC", "Fibonacci", "MalaysianSnR"],
            "TRENDING_BEAR": ["SMC", "Fibonacci", "MalaysianSnR"],
            "RANGING": ["LiquiditySweep", "MalaysianSnR"],
            "HIGH_VOLATILITY": [] # Safety first
        }

    def get_active_strategies(self, regime):
        """
        Returns a list of strategy instances that should be active for the given regime.
        """
        allowed_names = self.regime_map.get(regime, [])
        active_strategies = []

        for strategy in self.strategies:
            # Check if strategy name or class name is in the allowed list
            if strategy.name in allowed_names or type(strategy).__name__.replace("Strategy", "") in allowed_names:
                active_strategies.append(strategy)

        if not active_strategies and regime != "HIGH_VOLATILITY":
            logger.warning(f"No strategies configured for regime: {regime}")

        return active_strategies

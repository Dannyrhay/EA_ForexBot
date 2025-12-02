import logging
import pandas as pd
from .data_feed import DataFeed
from .execution import ExecutionEngine
from strategies.regime_filter import RegimeFilter
from strategies.strategy_selector import StrategySelector
from strategies.smc import SMCStrategy
# Import other strategies as needed

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, data, symbol, initial_balance=10000):
        self.data_feed = DataFeed(data, symbol)
        self.execution = ExecutionEngine(initial_balance)
        self.symbol = symbol

        # Initialize Components
        self.regime_filter = RegimeFilter()

        # Initialize Strategies
        # Note: In a real scenario, we'd load these from config
        self.strategies = [
            SMCStrategy("SMC"),
            # Add others
        ]
        self.strategy_selector = StrategySelector(self.strategies)

        self.history = [] # For equity curve

    def run(self):
        logger.info("Starting Event-Driven Backtest...")

        # Buffer to hold data for strategy analysis (need history for indicators)
        data_buffer = []

        for event in self.data_feed.get_next_event():
            # 1. Update Execution Engine with Tick
            if event['type'] == 'tick':
                self.execution.on_tick(event)

            # 2. Process Bar Close (Strategy Logic)
            elif event['type'] == 'bar_close':
                bar_data = event['data']
                data_buffer.append(bar_data)

                # Need enough data for indicators (e.g., 50 bars)
                if len(data_buffer) < 50:
                    continue

                # Create DataFrame for analysis
                df = pd.DataFrame(data_buffer)

                # A. Regime Detection
                regime = self.regime_filter.get_regime(df)

                # B. Strategy Selection
                active_strategies = self.strategy_selector.get_active_strategies(regime)

                # C. Signal Generation
                for strategy in active_strategies:
                    if isinstance(strategy, SMCStrategy):
                        setup = strategy.get_signal(df, symbol=self.symbol)

                        if setup and isinstance(setup, dict) and 'entry_price' in setup:
                            # Place Limit Order in Simulation
                            self.execution.place_limit_order(
                                symbol=self.symbol,
                                order_type=setup['type'],
                                price=setup['entry_price'],
                                sl=setup['sl'],
                                tp=setup['tp'],
                                volume=0.01,
                                comment="Backtest SMC"
                            )

        logger.info("Backtest Completed.")
        logger.info(f"Final Balance: {self.execution.balance}")
        return self.execution.trade_history

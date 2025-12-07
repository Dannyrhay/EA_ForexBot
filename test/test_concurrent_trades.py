import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock MetaTrader5 before importing main
sys.modules['MetaTrader5'] = MagicMock()
import MetaTrader5 as mt5

# Mock other dependencies
sys.modules['utils.db_connector'] = MagicMock()
sys.modules['utils.logging'] = MagicMock()
sys.modules['strategies.ml_model'] = MagicMock()
sys.modules['strategies.regime_filter'] = MagicMock()
sys.modules['strategies.strategy_selector'] = MagicMock()
sys.modules['strategies.sentiment_filter'] = MagicMock()
sys.modules['utils.order_manager'] = MagicMock()
sys.modules['utils.risk_manager'] = MagicMock()
sys.modules['utils.mt5_connection'] = MagicMock()
sys.modules['utils.feature_engineering'] = MagicMock()
sys.modules['utils.trade_history'] = MagicMock()

# Define dummy classes for isinstance checks
class MockStrategy:
    def __init__(self, name="Mock", **kwargs): self.name = name
    def set_config(self, cfg): pass

class MockADXStrategy(MockStrategy): pass
class MockSMCStrategy(MockStrategy): pass
class MockLiquiditySweepStrategy(MockStrategy): pass
class MockFibonacciStrategy(MockStrategy): pass
class MockMalaysianSnRStrategy(MockStrategy): pass

# Mock strategy modules to return these classes
mock_adx_module = MagicMock()
mock_adx_module.ADXStrategy = MockADXStrategy
sys.modules['strategies.adx_strategy'] = mock_adx_module

mock_smc_module = MagicMock()
mock_smc_module.SMCStrategy = MockSMCStrategy
sys.modules['strategies.smc'] = mock_smc_module

mock_ls_module = MagicMock()
mock_ls_module.LiquiditySweepStrategy = MockLiquiditySweepStrategy
sys.modules['strategies.liquidity_sweep'] = mock_ls_module

mock_fib_module = MagicMock()
mock_fib_module.FibonacciStrategy = MockFibonacciStrategy
sys.modules['strategies.fibonacci'] = mock_fib_module

mock_snr_module = MagicMock()
mock_snr_module.MalaysianSnRStrategy = MockMalaysianSnRStrategy
sys.modules['strategies.malaysian_snr'] = mock_snr_module

sys.modules['consensus'] = MagicMock()

from main import TradingBot, ADXStrategy # Import ADXStrategy to patch it if needed, but main imports it from strategies.adx_strategy

class TestConcurrentTrades(unittest.TestCase):
    def setUp(self):
        self.bot = TradingBot()
        self.bot.config = {'max_open_positions_per_symbol': 5}

    def test_can_open_new_trade_no_positions(self):
        # Setup: No open positions
        mt5.positions_get.return_value = []
        
        # Test
        result = self.bot.can_open_new_trade('EURUSD', 'StrategyA')
        self.assertTrue(result)

    def test_can_open_new_trade_different_strategy(self):
        # Setup: One open position from StrategyA
        mock_pos = MagicMock()
        mock_pos.comment = "AI-StrategyA-H1"
        mt5.positions_get.return_value = [mock_pos]
        
        # Test: Can StrategyB open a trade?
        result = self.bot.can_open_new_trade('EURUSD', 'StrategyB')
        self.assertTrue(result)

    def test_can_open_new_trade_same_strategy(self):
        # Setup: One open position from StrategyA
        mock_pos = MagicMock()
        mock_pos.comment = "AI-StrategyA-H1"
        mt5.positions_get.return_value = [mock_pos]
        
        # Test: Can StrategyA open another trade?
        result = self.bot.can_open_new_trade('EURUSD', 'StrategyA')
        self.assertFalse(result)

    def test_can_open_new_trade_global_cap(self):
        # Setup: Max positions reached (5 different strategies)
        positions = []
        for i in range(5):
            p = MagicMock()
            p.comment = f"AI-Strategy{i}-H1"
            positions.append(p)
        mt5.positions_get.return_value = positions
        
        # Test: Can StrategyNew open a trade?
        result = self.bot.can_open_new_trade('EURUSD', 'StrategyNew')
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()

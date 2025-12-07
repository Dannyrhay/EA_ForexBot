import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import time

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
sys.modules['utils.mt5_connection'] = MagicMock()
sys.modules['utils.feature_engineering'] = MagicMock()
sys.modules['utils.trade_history'] = MagicMock()
sys.modules['consensus'] = MagicMock()

# Mock strategies
class MockStrategy:
    def __init__(self, name="Mock", **kwargs): self.name = name
    def set_config(self, cfg): pass
class MockADXStrategy(MockStrategy): pass
class MockSMCStrategy(MockStrategy): pass
class MockLiquiditySweepStrategy(MockStrategy): pass
class MockFibonacciStrategy(MockStrategy): pass
class MockMalaysianSnRStrategy(MockStrategy): pass

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

from utils.risk_manager import RiskManager

class TestRiskManagerThrottling(unittest.TestCase):
    def setUp(self):
        self.config = {
            'portfolio_risk': {
                'enabled': True,
                'max_daily_drawdown_percent': 5.0,
                'max_portfolio_risk_percent': 2.0,
                'max_margin_level_percent': 10.0
            }
        }
        self.risk_manager = RiskManager(self.config)

    def test_should_log(self):
        # First call should return True
        self.assertTrue(self.risk_manager._should_log("test_key", interval=1))
        
        # Immediate second call should return False
        self.assertFalse(self.risk_manager._should_log("test_key", interval=1))
        
        # Wait for interval
        time.sleep(1.1)
        
        # Third call should return True
        self.assertTrue(self.risk_manager._should_log("test_key", interval=1))

if __name__ == '__main__':
    unittest.main()

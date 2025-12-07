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
sys.modules['utils.risk_manager'] = MagicMock() # We will use the real one or mock specifically
sys.modules['utils.mt5_connection'] = MagicMock()
sys.modules['utils.feature_engineering'] = MagicMock()
sys.modules['utils.trade_history'] = MagicMock()

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

sys.modules['consensus'] = MagicMock()

from main import TradingBot
from utils.risk_manager import RiskManager

class TestPortfolioRisk(unittest.TestCase):
    def setUp(self):
        self.bot = TradingBot()
        # Enable portfolio risk in config
        self.bot.config['portfolio_risk'] = {
            'enabled': True,
            'max_daily_drawdown_percent': 5.0,
            'max_portfolio_risk_percent': 2.0,
            'max_margin_level_percent': 10.0
        }
        # Use real RiskManager with this config
        self.bot.risk_manager = RiskManager(self.bot.config)
        self.bot.daily_starting_equity = 10000.0

    def test_daily_drawdown_check(self):
        # Setup: Current equity dropped to 9400 (6% loss)
        mock_acc = MagicMock()
        mock_acc.equity = 9400.0
        mock_acc.margin = 100.0
        mt5.account_info.return_value = mock_acc
        
        # Test
        result = self.bot.can_open_new_trade('EURUSD')
        self.assertFalse(result, "Should block trade due to 6% drawdown")

    def test_margin_level_check(self):
        # Setup: Equity 10000, Margin 1100 (11% used)
        mock_acc = MagicMock()
        mock_acc.equity = 10000.0
        mock_acc.margin = 1100.0
        mt5.account_info.return_value = mock_acc
        
        # Test
        result = self.bot.can_open_new_trade('EURUSD')
        self.assertFalse(result, "Should block trade due to 11% margin usage")

    def test_portfolio_risk_check(self):
        # Setup: Equity 10000. Max risk 2% = 200.
        # Existing trade: Risk 250 (2.5%)
        mock_acc = MagicMock()
        mock_acc.equity = 10000.0
        mock_acc.margin = 100.0
        mt5.account_info.return_value = mock_acc
        
        mock_pos = MagicMock()
        mock_pos.symbol = 'EURUSD'
        mock_pos.volume = 1.0
        mock_pos.price_open = 1.1000
        mock_pos.sl = 1.0975 # 25 pips
        
        # Mock symbol info for risk calc
        mock_symbol = MagicMock()
        mock_symbol.trade_contract_size = 100000
        mt5.symbol_info.return_value = mock_symbol
        
        mt5.positions_get.return_value = [mock_pos]
        
        # Test
        result = self.bot.can_open_new_trade('GBPUSD')
        self.assertFalse(result, "Should block trade due to excessive portfolio risk")

if __name__ == '__main__':
    unittest.main()

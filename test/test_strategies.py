import unittest
import pandas as pd
from strategies.supply_demand import SupplyDemandStrategy

class TestStrategies(unittest.TestCase):
    def test_supply_demand(self):
        data = pd.DataFrame({
            'low': [1.0, 1.1, 0.9, 1.2],
            'high': [1.2, 1.3, 1.1, 1.4],
            'close': [1.1, 1.2, 1.0, 1.3]
        })
        strategy = SupplyDemandStrategy(window=2)
        signal = strategy.get_signal(data)
        self.assertEqual(signal, 'buy')

if __name__ == '__main__':
    unittest.main()
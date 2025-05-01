from .base_strategy import BaseStrategy
import pandas as pd

class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy: Generates trading signals based on price touching the Bollinger Bands.
    
    - Sell when price touches or exceeds the upper band.
    - Buy when price touches or goes below the lower band.
    - Hold otherwise.
    
    Parameters:
    - window: int, number of periods for the moving average (default=20)
    - std_dev: float, number of standard deviations for the bands (default=2.0)
    """
    def __init__(self, name, window=20, std_dev=2.0):
        super().__init__(name)
        self.window = window
        self.std_dev = std_dev

    def get_signal(self, data, symbol=None, timeframe=None):
        """
        Generate trading signal based on Bollinger Bands.
        
        Parameters:
        - data: DataFrame with 'close' column
        - symbol: str, trading symbol (optional)
        - timeframe: str, timeframe of the data (optional)
        
        Returns:
        - tuple: (signal, strength) where signal is 'buy', 'sell', or 'hold', and strength is a float
        """
        try:
            df = pd.DataFrame(data)
            if not all(col in df for col in ['close']):
                print(f"BollingerBands: Missing required columns for {symbol} on {timeframe}")
                return ('hold', 0.0)
            ma = df['close'].rolling(window=self.window).mean()
            std = df['close'].rolling(window=self.window).std()
            upper_band = ma + self.std_dev * std
            lower_band = ma - self.std_dev * std
            current_close = df['close'].iloc[-1]
            if current_close >= upper_band.iloc[-1]:
                return ('sell', 0.8)
            elif current_close <= lower_band.iloc[-1]:
                return ('buy', 0.8)
            else:
                return ('hold', 0.0)
        except Exception as e:
            print(f"BollingerBands: Error in get_signal for {symbol} on {timeframe}: {e}")
            return ('hold', 0.0)
import MetaTrader5 as mt5
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Centralized risk management module for calculating SL/TP and position sizing.
    Supports ATR-based, percentage-based, and fixed pips methods.
    """

    def __init__(self, config):
        self.config = config
        self.risk_config = config.get('risk_management', {})
        self.method = self.risk_config.get('method', 'atr').lower()
        logger.info(f"RiskManager initialized with method: {self.method}")

    def calculate_atr(self, data, period=14):
        """Calculate Average True Range from OHLC data."""
        if not isinstance(data, pd.DataFrame) or data.empty:
            return None

        if len(data) < period:
            logger.warning(f"Insufficient data for ATR calculation: {len(data)} < {period}")
            return None

        try:
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]

            return float(atr) if pd.notna(atr) else None
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return None

    def calculate_sl_tp_atr(self, symbol, entry_price, direction, data):
        """
        Calculate SL and TP using ATR-based method.

        Args:
            symbol: Trading symbol
            entry_price: Entry price for the trade
            direction: 'buy' or 'sell'
            data: OHLC DataFrame for ATR calculation

        Returns:
            (sl_price, tp_price) tuple
        """
        atr_params = self.risk_config.get('atr_params', {})
        sl_multiplier = atr_params.get('sl_multiplier', 1.5)
        tp_rr_ratio = atr_params.get('tp_risk_reward_ratio', 1.5)
        atr_period = atr_params.get('atr_period', 14)

        atr = self.calculate_atr(data, period=atr_period)

        if atr is None or atr == 0:
            logger.warning(f"Invalid ATR for {symbol}, using fallback percentage method")
            return self.calculate_sl_tp_percentage(symbol, entry_price, direction)

        sl_distance = atr * sl_multiplier
        tp_distance = sl_distance * tp_rr_ratio

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol info not found for {symbol}")
            return None, None

        digits = symbol_info.digits

        if direction.lower() == 'buy':
            sl_price = round(entry_price - sl_distance, digits)
            tp_price = round(entry_price + tp_distance, digits)
        else:  # sell
            sl_price = round(entry_price + sl_distance, digits)
            tp_price = round(entry_price - tp_distance, digits)

        logger.info(f"ATR-based SL/TP for {symbol} {direction.upper()}: ATR={atr:.5f}, SL={sl_price}, TP={tp_price}")

        return sl_price, tp_price

    def calculate_sl_tp_percentage(self, symbol, entry_price, direction):
        """
        Calculate SL and TP using percentage-based method.

        Args:
            symbol: Trading symbol
            entry_price: Entry price for the trade
            direction: 'buy' or 'sell'

        Returns:
            (sl_price, tp_price) tuple
        """
        pct_params = self.risk_config.get('percentage_params', {})
        sl_percent = pct_params.get('sl_percent', 1.0) / 100.0
        tp_percent = pct_params.get('tp_percent', 1.5) / 100.0

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol info not found for {symbol}")
            return None, None

        digits = symbol_info.digits

        if direction.lower() == 'buy':
            sl_price = round(entry_price * (1 - sl_percent), digits)
            tp_price = round(entry_price * (1 + tp_percent), digits)
        else:  # sell
            sl_price = round(entry_price * (1 + sl_percent), digits)
            tp_price = round(entry_price * (1 - tp_percent), digits)

        logger.info(f"Percentage-based SL/TP for {symbol} {direction.upper()}: SL={sl_price}, TP={tp_price}")

        return sl_price, tp_price

    def calculate_sl_tp_fixed_pips(self, symbol, entry_price, direction):
        """
        Calculate SL and TP using fixed pips method.

        Args:
            symbol: Trading symbol
            entry_price: Entry price for the trade
            direction: 'buy' or 'sell'

        Returns:
            (sl_price, tp_price) tuple
        """
        pips_params = self.risk_config.get('fixed_pips_params', {})
        sl_pips = pips_params.get('sl_pips', 20)
        tp_pips = pips_params.get('tp_pips', 30)

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol info not found for {symbol}")
            return None, None

        digits = symbol_info.digits
        point = symbol_info.point

        # Convert pips to price distance
        pip_value = point * 10 if digits == 5 or digits == 3 else point
        sl_distance = sl_pips * pip_value
        tp_distance = tp_pips * pip_value

        if direction.lower() == 'buy':
            sl_price = round(entry_price - sl_distance, digits)
            tp_price = round(entry_price + tp_distance, digits)
        else:  # sell
            sl_price = round(entry_price + sl_distance, digits)
            tp_price = round(entry_price - tp_distance, digits)

        logger.info(f"Fixed pips SL/TP for {symbol} {direction.upper()}: SL={sl_price}, TP={tp_price}")

        return sl_price, tp_price

    def calculate_sl_tp(self, symbol, entry_price, direction, data=None):
        """
        Main method to calculate SL/TP using configured method.

        Args:
            symbol: Trading symbol
            entry_price: Entry price for the trade
            direction: 'buy' or 'sell'
            data: OHLC DataFrame (required for ATR method)

        Returns:
            (sl_price, tp_price) tuple
        """
        if not self.risk_config.get('enabled', True):
            logger.info("Risk management disabled in config")
            return None, None

        if self.method == 'atr':
            if data is None:
                logger.warning("ATR method requires data, falling back to percentage")
                return self.calculate_sl_tp_percentage(symbol, entry_price, direction)
            return self.calculate_sl_tp_atr(symbol, entry_price, direction, data)

        elif self.method == 'percentage':
            return self.calculate_sl_tp_percentage(symbol, entry_price, direction)

        elif self.method == 'fixed_pips':
            return self.calculate_sl_tp_fixed_pips(symbol, entry_price, direction)

        else:
            logger.warning(f"Unknown risk management method: {self.method}, using percentage")
            return self.calculate_sl_tp_percentage(symbol, entry_price, direction)

    def calculate_position_size(self, symbol, account_balance, sl_distance):
        """
        Calculate position size based on risk percentage.

        Args:
            symbol: Trading symbol
            account_balance: Current account balance
            sl_distance: Distance to stop loss in price units

        Returns:
            Position size in lots
        """
        max_risk_percent = self.risk_config.get('max_risk_percent_per_trade', 2.0) / 100.0
        risk_amount = account_balance * max_risk_percent

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol info not found for {symbol}")
            return symbol_info.volume_min if symbol_info else 0.01

        contract_size = symbol_info.trade_contract_size
        point = symbol_info.point

        # Calculate value per point
        value_per_point = contract_size * point

        # Calculate position size
        sl_distance_points = sl_distance / point
        position_size = risk_amount / (sl_distance_points * value_per_point)

        # Round to min step and apply limits
        volume_step = symbol_info.volume_step
        position_size = round(position_size / volume_step) * volume_step
        position_size = max(symbol_info.volume_min, min(position_size, symbol_info.volume_max))

        logger.info(f"Calculated position size for {symbol}: {position_size} lots (Risk: {max_risk_percent*100}%)")

        return position_size

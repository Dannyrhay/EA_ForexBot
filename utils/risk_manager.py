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
<<<<<<< HEAD
        self._last_log_time = {} # [NEW] For throttling logs
        logger.info(f"RiskManager initialized with method: {self.method}")

    # ... (existing methods) ...

    def _should_log(self, key, interval=60):
        """Helper to check if we should log based on interval."""
        import time
        current_time = time.time()
        last_time = self._last_log_time.get(key, 0)
        if current_time - last_time > interval:
            self._last_log_time[key] = current_time
            return True
        return False

    def calculate_atr(self, data, period=14):
        # ... (unchanged) ...
=======
        logger.info(f"RiskManager initialized with method: {self.method}")

    def calculate_atr(self, data, period=14):
>>>>>>> 3bf0cf4babc04168161ee0889422e8e811a2ac82
        """Calculate Average True Range from OHLC data."""
        if not isinstance(data, pd.DataFrame) or data.empty:
            return None

        if len(data) < period:
<<<<<<< HEAD
            # Throttle this warning too
            if self._should_log(f"atr_insufficient_{len(data)}"):
                logger.warning(f"Insufficient data for ATR calculation: {len(data)} < {period}")
=======
            logger.warning(f"Insufficient data for ATR calculation: {len(data)} < {period}")
>>>>>>> 3bf0cf4babc04168161ee0889422e8e811a2ac82
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

<<<<<<< HEAD
    # ... (calculate_sl_tp methods unchanged) ...

    def check_daily_drawdown(self, current_equity, daily_starting_equity):
        """
        Checks if the account has hit the maximum daily drawdown limit.
        Returns False if trading should stop.
        """
        if not self.config.get('portfolio_risk', {}).get('enabled', False):
            return True

        max_dd_percent = self.config.get('portfolio_risk', {}).get('max_daily_drawdown_percent', 5.0)
        
        if daily_starting_equity <= 0: return True # Avoid division by zero

        drawdown = (daily_starting_equity - current_equity) / daily_starting_equity * 100
        
        if drawdown >= max_dd_percent:
            if self._should_log("daily_drawdown"):
                logger.warning(f"Daily Drawdown Limit Hit! Drawdown: {drawdown:.2f}% >= Limit: {max_dd_percent}%")
            return False
        
        return True

    def check_portfolio_risk(self, open_positions, current_equity, new_trade_risk_amount=0):
        """
        Checks if the total risk of all open positions + new trade exceeds the portfolio limit.
        Returns False if risk is too high.
        """
        if not self.config.get('portfolio_risk', {}).get('enabled', False):
            return True

        max_portfolio_risk_percent = self.config.get('portfolio_risk', {}).get('max_portfolio_risk_percent', 2.0)
        max_risk_amount = current_equity * (max_portfolio_risk_percent / 100.0)

        current_risk_exposure = 0.0
        
        # Calculate risk of existing positions
        if open_positions:
            for pos in open_positions:
                # Estimate risk based on SL distance
                # If no SL, assume full value risk (or skip, but safer to assume risk)
                # Here we try to calculate monetary risk if SL exists
                if pos.sl > 0:
                    symbol_info = mt5.symbol_info(pos.symbol)
                    if symbol_info:
                        price_diff = abs(pos.price_open - pos.sl)
                        # Value = Volume * ContractSize * PriceDiff
                        # For Forex, Profit = (Close - Open) * Volume * ContractSize
                        # This is an approximation. 
                        # Better: use OrderCalcProfit but that requires async/tick data
                        # Simple approx:
                        risk = price_diff * pos.volume * symbol_info.trade_contract_size
                        # Adjust for currency conversion if needed (assuming USD account and USD quote)
                        # If quote currency is not account currency, we need conversion.
                        # For simplicity/speed, we assume USD account or close enough for now.
                        current_risk_exposure += risk
                else:
                    # If no SL, this is high risk. Maybe count a default %?
                    pass

        total_risk = current_risk_exposure + new_trade_risk_amount

        if total_risk > max_risk_amount:
            if self._should_log("portfolio_risk"):
                logger.warning(f"Max Portfolio Risk Exceeded! Total Risk: {total_risk:.2f} > Limit: {max_risk_amount:.2f} ({max_portfolio_risk_percent}%)")
            return False

        return True

    def check_margin_level(self, account_info):
        """
        Checks if the account margin usage is within safe limits.
        Returns False if margin usage is too high (margin level too low).
        Note: Config uses 'max_margin_level_percent' which implies 'used margin %'.
        MT5 gives 'margin_level' which is Equity / Margin * 100.
        So if max used margin is 10%, that means min margin level is 1000%.
        Let's interpret config as 'Max Used Margin % of Equity'.
        Used Margin % = (Margin / Equity) * 100
        """
        if not self.config.get('portfolio_risk', {}).get('enabled', False):
            return True

        max_used_margin_percent = self.config.get('portfolio_risk', {}).get('max_margin_level_percent', 10.0)
        
        if account_info.equity <= 0: return False

        used_margin_percent = (account_info.margin / account_info.equity) * 100
        
        if used_margin_percent >= max_used_margin_percent:
            if self._should_log("margin_level"):
                logger.warning(f"Margin Usage Limit Hit! Used: {used_margin_percent:.2f}% >= Limit: {max_used_margin_percent}%")
            return False

        return True

=======
>>>>>>> 3bf0cf4babc04168161ee0889422e8e811a2ac82
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
<<<<<<< HEAD

=======
>>>>>>> 3bf0cf4babc04168161ee0889422e8e811a2ac82

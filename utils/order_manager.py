import MetaTrader5 as mt5
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class OrderManager:
    """
    Handles the placement, modification, and cancellation of pending orders (Limit/Stop).
    Ensures precision execution and manages order lifecycle.
    """
    def __init__(self, magic_number=123456):
        self.magic_number = magic_number

    def place_limit_order(self, symbol, order_type, price, sl, tp, volume, comment="Limit Order", expiration=None):
        """
        Places a Limit Order at a specific price.
        """
        if not mt5.terminal_info():
            logger.error("MT5 not connected. Cannot place order.")
            return None

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol {symbol} not found.")
            return None

        # Normalize price, SL, TP
        price = round(price, symbol_info.digits)
        sl = round(sl, symbol_info.digits) if sl else 0.0
        tp = round(tp, symbol_info.digits) if tp else 0.0

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC, # Good Till Cancelled
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        if expiration:
            request["type_time"] = mt5.ORDER_TIME_SPECIFIED
            request["expiration"] = expiration

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to place limit order for {symbol}: {result.comment} ({result.retcode})")
            return None

        logger.info(f"Limit Order Placed: {symbol} {order_type} @ {price}, Vol: {volume}, Ticket: {result.order}")
        return result.order

    def modify_order(self, ticket, price=None, sl=None, tp=None):
        """
        Modifies an existing pending order.
        """
        # Logic to modify order
        pass # To be implemented if needed for trailing entries

    def cancel_order(self, ticket):
        """
        Cancels a pending order.
        """
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
            "magic": self.magic_number,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to cancel order {ticket}: {result.comment} ({result.retcode})")
            return False

        logger.info(f"Order {ticket} cancelled successfully.")
        return True

    def get_pending_orders(self, symbol=None):
        """
        Returns a list of pending orders, optionally filtered by symbol.
        """
        if symbol:
            orders = mt5.orders_get(symbol=symbol)
        else:
            orders = mt5.orders_get()

        if orders is None:
            return []

        # Filter by magic number
        return [o for o in orders if o.magic == self.magic_number]

    def place_market_order(self, symbol, order_type, volume, sl=None, tp=None, comment="Market Order"):
        """
        Places a Market Order (Buy/Sell) immediately.
        """
        logger.info(f"DEBUG: place_market_order called for {symbol} {order_type} vol={volume} sl={sl} tp={tp}")

        if not mt5.terminal_info():
            logger.error("MT5 not connected. Cannot place order.")
            return None

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol {symbol} not found.")
            return None

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Symbol {symbol} not found or cannot be selected.")
                return None

        # Normalize SL, TP
        sl = round(sl, symbol_info.digits) if sl else 0.0
        tp = round(tp, symbol_info.digits) if tp else 0.0

        # Determine price
        price = symbol_info.ask if order_type == mt5.ORDER_TYPE_BUY else symbol_info.bid

        # Determine filling mode dynamically
        # Define constants locally as they might be missing in some mt5 package versions
        SYMBOL_FILLING_FOK = 1
        SYMBOL_FILLING_IOC = 2

        filling_mode = mt5.ORDER_FILLING_FOK # Default fallback

        if symbol_info.filling_mode & SYMBOL_FILLING_IOC:
            filling_mode = mt5.ORDER_FILLING_IOC
        elif symbol_info.filling_mode & SYMBOL_FILLING_FOK:
            filling_mode = mt5.ORDER_FILLING_FOK
        else:
            # If neither FOK nor IOC is explicitly flagged, assume RETURN (common for Market Execution)
            filling_mode = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        logger.info(f"DEBUG: Sending order request: {request}")
        result = mt5.order_send(request)
        logger.info(f"DEBUG: Order result: {result}")

        if result is None:
             logger.error("MT5 order_send returned None!")
             return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to place market order for {symbol}: {result.comment} ({result.retcode})")
            return None

        logger.info(f"Market Order Placed: {symbol} {order_type} @ {result.price}, Vol: {volume}, Ticket: {result.order}")
        return result.order

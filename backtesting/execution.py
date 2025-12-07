import logging

logger = logging.getLogger(__name__)

class ExecutionEngine:
    """
    Simulates a Broker. Handles Order Placement, Execution, and Position Management.
    """
    def __init__(self, initial_balance=10000, spread=0.0001):
        self.balance = initial_balance
        self.equity = initial_balance
        self.spread = spread
        self.pending_orders = {} # {ticket: {order_details}}
        self.open_positions = {} # {ticket: {position_details}}
        self.trade_history = []
        self.ticket_counter = 1

    def place_limit_order(self, symbol, order_type, price, sl, tp, volume, comment):
        ticket = self.ticket_counter
        self.ticket_counter += 1

        self.pending_orders[ticket] = {
            'ticket': ticket,
            'symbol': symbol,
            'type': order_type, # 2=Buy Limit, 3=Sell Limit (using MT5 constants logic)
            'price': price,
            'sl': sl,
            'tp': tp,
            'volume': volume,
            'comment': comment,
            'status': 'pending'
        }
        logger.info(f"[BACKTEST] Order Placed: {self.pending_orders[ticket]}")
        return ticket

    def on_tick(self, tick):
        """
        Process a new tick: Check for fills and SL/TP hits.
        """
        price = tick['price']

        # 1. Check Pending Orders
        # We need to iterate over a copy because we might modify the dict
        for ticket, order in list(self.pending_orders.items()):
            if order['symbol'] != tick['symbol']: continue

            # Buy Limit: Price drops below limit price
            # MT5 ORDER_TYPE_BUY_LIMIT = 2
            if order['type'] == 2:
                # Add spread to Ask price for Buy orders
                ask_price = price + self.spread
                if ask_price <= order['price']:
                    self._fill_order(ticket, ask_price, tick['time'])

            # Sell Limit: Price rises above limit price
            # MT5 ORDER_TYPE_SELL_LIMIT = 3
            elif order['type'] == 3:
                bid_price = price
                if bid_price >= order['price']:
                    self._fill_order(ticket, bid_price, tick['time'])

        # 2. Check Open Positions (SL/TP)
        for ticket, pos in list(self.open_positions.items()):
            if pos['symbol'] != tick['symbol']: continue

            # Buy Position
            if pos['type'] == 'BUY':
                bid_price = price
                # Check SL
                if pos['sl'] > 0 and bid_price <= pos['sl']:
                    self._close_position(ticket, bid_price, tick['time'], 'sl')
                # Check TP
                elif pos['tp'] > 0 and bid_price >= pos['tp']:
                    self._close_position(ticket, bid_price, tick['time'], 'tp')

            # Sell Position
            elif pos['type'] == 'SELL':
                ask_price = price + self.spread
                # Check SL
                if pos['sl'] > 0 and ask_price >= pos['sl']:
                    self._close_position(ticket, ask_price, tick['time'], 'sl')
                # Check TP
                elif pos['tp'] > 0 and ask_price <= pos['tp']:
                    self._close_position(ticket, ask_price, tick['time'], 'tp')

    def _fill_order(self, ticket, fill_price, time):
        order = self.pending_orders.pop(ticket)

        position = {
            'ticket': ticket,
            'symbol': order['symbol'],
            'type': 'BUY' if order['type'] == 2 else 'SELL',
            'entry_price': fill_price,
            'volume': order['volume'],
            'sl': order['sl'],
            'tp': order['tp'],
            'open_time': time,
            'comment': order['comment']
        }
        self.open_positions[ticket] = position
        logger.info(f"[BACKTEST] Order Filled: {ticket} @ {fill_price}")

    def _close_position(self, ticket, close_price, time, reason):
        pos = self.open_positions.pop(ticket)

        profit = 0
        if pos['type'] == 'BUY':
            profit = (close_price - pos['entry_price']) * pos['volume'] * 100000 # Approx for Forex
        else:
            profit = (pos['entry_price'] - close_price) * pos['volume'] * 100000

        self.balance += profit

        trade_record = {
            **pos,
            'close_price': close_price,
            'close_time': time,
            'profit': profit,
            'reason': reason
        }
        self.trade_history.append(trade_record)
        logger.info(f"[BACKTEST] Position Closed: {ticket} | P/L: {profit:.2f} | Reason: {reason}")

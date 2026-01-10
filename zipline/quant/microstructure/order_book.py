"""Limit Order Book Simulation."""
import numpy as np
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

class LimitOrderBook:
    """Full limit order book simulation."""
    
    def __init__(self):
        self.bids = defaultdict(list)  # price -> [orders]
        self.asks = defaultdict(list)
        logger.info("LimitOrderBook initialized")
    
    def add_order(self, side, price, quantity):
        """Add limit order to book."""
        if side == 'buy':
            self.bids[price].append(quantity)
        else:
            self.asks[price].append(quantity)
    
    def get_best_bid(self):
        """Get best bid price."""
        return max(self.bids.keys()) if self.bids else None
    
    def get_best_ask(self):
        """Get best ask price."""
        return min(self.asks.keys()) if self.asks else None
    
    def get_spread(self):
        """Calculate bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        return best_ask - best_bid if best_bid and best_ask else None

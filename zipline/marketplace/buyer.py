"""Buyer Profile Management"""
from typing import List, Dict, Any
from datetime import datetime


class BuyerProfile:
    """Buyer profile management."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.purchased_strategies: List[str] = []
        self.subscriptions: List[Dict[str, Any]] = []
        self.purchase_history: List[Dict[str, Any]] = []
    
    def add_purchase(self, listing_id: str, amount: float) -> None:
        """Record a purchase."""
        self.purchased_strategies.append(listing_id)
        self.purchase_history.append({
            'listing_id': listing_id,
            'amount': amount,
            'date': datetime.now().isoformat(),
        })

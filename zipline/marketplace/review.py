"""Strategy Review System"""
from typing import Optional
from datetime import datetime


class StrategyReview:
    """User reviews and ratings for strategies."""
    
    def __init__(self, listing_id: str, user_id: str, rating: float, comment: str = ""):
        self.listing_id = listing_id
        self.user_id = user_id
        self.rating = rating  # 1-5
        self.comment = comment
        self.created_at = datetime.now()
        self.helpful_votes = 0
        self.verified_purchase = False
    
    def add_helpful_vote(self) -> None:
        """Add a helpful vote."""
        self.helpful_votes += 1

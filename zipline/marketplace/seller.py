"""Seller Profile Management"""
from typing import Dict, List, Any
from datetime import datetime


class SellerProfile:
    """Seller profile and reputation management."""
    
    def __init__(self, user_id: str, display_name: str):
        self.user_id = user_id
        self.display_name = display_name
        self.verification_level = "unverified"
        self.total_sales = 0
        self.total_revenue = 0.0
        self.ratings: List[float] = []
        self.listings: List[str] = []
    
    def add_sale(self, amount: float) -> None:
        """Record a sale."""
        self.total_sales += 1
        self.total_revenue += amount
    
    def add_rating(self, rating: float) -> None:
        """Add a rating."""
        self.ratings.append(rating)
    
    def get_average_rating(self) -> float:
        """Get average rating."""
        return sum(self.ratings) / len(self.ratings) if self.ratings else 0.0

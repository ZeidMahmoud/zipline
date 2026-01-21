"""
Strategy Marketplace Platform
"""
from typing import Dict, List, Optional, Any
from datetime import datetime


class StrategyMarketplace:
    """
    Main marketplace platform for strategy discovery and trading.
    
    Parameters
    ----------
    name : str
        Marketplace name
    commission_rate : float, optional
        Commission rate (0-1) on transactions
        
    Examples
    --------
    >>> marketplace = StrategyMarketplace("Zipline Strategies")
    >>> listings = marketplace.search_strategies(category="momentum")
    """
    
    def __init__(self, name: str, commission_rate: float = 0.10):
        self.name = name
        self.commission_rate = commission_rate
        self.listings: Dict[str, Any] = {}
        self._next_id = 1
    
    def list_strategy(self, listing_data: Dict[str, Any]) -> str:
        """Create a new strategy listing."""
        listing_id = f"listing_{self._next_id}"
        self._next_id += 1
        
        self.listings[listing_id] = {
            'id': listing_id,
            'created_at': datetime.now().isoformat(),
            **listing_data
        }
        
        return listing_id
    
    def search_strategies(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        min_rating: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for strategies."""
        results = list(self.listings.values())
        
        if category:
            results = [r for r in results if r.get('category') == category]
        if min_rating:
            results = [r for r in results if r.get('rating', 0) >= min_rating]
        if max_price:
            results = [r for r in results if r.get('price', float('inf')) <= max_price]
        
        return results
    
    def get_listing(self, listing_id: str) -> Optional[Dict[str, Any]]:
        """Get strategy listing details."""
        return self.listings.get(listing_id)
    
    def get_featured_strategies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get featured strategies."""
        featured = [l for l in self.listings.values() if l.get('featured', False)]
        return featured[:limit]

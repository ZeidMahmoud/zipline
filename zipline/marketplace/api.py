"""Marketplace REST API"""
from typing import Dict, Any, Optional


class MarketplaceAPI:
    """REST API interface for marketplace."""
    
    def __init__(self, marketplace):
        self.marketplace = marketplace
    
    def list_strategies(
        self,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List strategies."""
        strategies = self.marketplace.search_strategies(category=category)
        return {'strategies': strategies[:limit], 'total': len(strategies)}
    
    def get_strategy(self, listing_id: str) -> Dict[str, Any]:
        """Get strategy details."""
        listing = self.marketplace.get_listing(listing_id)
        if not listing:
            return {'error': 'Not found', 'status': 404}
        return {'listing': listing, 'status': 200}

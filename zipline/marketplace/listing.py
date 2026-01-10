"""Strategy Listing Management"""
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime


class PricingModel(Enum):
    """Pricing models for strategies"""
    ONE_TIME = "one_time"
    SUBSCRIPTION = "subscription"
    REVENUE_SHARE = "revenue_share"
    FREE = "free"


class LicenseType(Enum):
    """License types"""
    PERSONAL = "personal"
    COMMERCIAL = "commercial"
    ENTERPRISE = "enterprise"


class StrategyListing:
    """Individual strategy listing."""
    
    def __init__(self, title: str, description: str, seller_id: str,
                 pricing_model: PricingModel, price: float = 0.0):
        self.title = title
        self.description = description
        self.seller_id = seller_id
        self.pricing_model = pricing_model
        self.price = price
        self.created_at = datetime.now()
        self.performance_metrics: Dict[str, float] = {}
        self.versions: List[str] = []
    
    def update_performance(self, metrics: Dict[str, float]) -> None:
        """Update performance metrics."""
        self.performance_metrics.update(metrics)
    
    def add_version(self, version: str) -> None:
        """Add a new version."""
        self.versions.append(version)

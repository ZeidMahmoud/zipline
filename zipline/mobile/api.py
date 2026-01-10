"""Mobile API optimizations"""


class MobileAPI:
    """Optimized API for mobile devices."""
    
    def __init__(self):
        self.compression_enabled = True
    
    def get_portfolio_summary(self, user_id: str) -> dict:
        """Get compressed portfolio summary."""
        return {'user_id': user_id, 'compressed': True}

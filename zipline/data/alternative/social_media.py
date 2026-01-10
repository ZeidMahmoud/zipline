"""Social Media Signals."""
import logging
logger = logging.getLogger(__name__)

class TwitterSentiment:
    """Twitter/X sentiment analysis."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
    
    def get_sentiment(self, symbol):
        """Get Twitter sentiment for symbol."""
        return 0.0

class RedditWallStreetBets:
    """Reddit WallStreetBets mention tracking."""
    
    def __init__(self):
        pass
    
    def get_mentions(self, symbol):
        """Get mention count for symbol."""
        return 0

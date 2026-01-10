"""Alternative Data Sources."""
try:
    from .sentiment import SentimentAnalyzer, FinBERTSentiment
    from .news import NewsDataLoader, NewsEventDetector
    from .social_media import TwitterSentiment, RedditWallStreetBets
    from .economic import FREDDataLoader, EconomicCalendar
    __all__ = ['SentimentAnalyzer', 'FinBERTSentiment', 'NewsDataLoader',
               'NewsEventDetector', 'TwitterSentiment', 'RedditWallStreetBets',
               'FREDDataLoader', 'EconomicCalendar']
except ImportError:
    __all__ = []

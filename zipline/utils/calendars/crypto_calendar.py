"""
24/7 Trading calendar for cryptocurrency markets.

This module provides a trading calendar for cryptocurrency exchanges
that operate continuously (24 hours a day, 7 days a week, 365 days a year).
"""
import pandas as pd
from datetime import time
from pytz import timezone

try:
    from trading_calendars import TradingCalendar
    from trading_calendars.trading_calendar import HolidayCalendar
except ImportError:
    # Fallback for development
    TradingCalendar = object
    HolidayCalendar = object


class CryptoCalendar(TradingCalendar):
    """
    Trading calendar for cryptocurrency markets.
    
    Crypto markets are open 24/7/365, so this calendar has no holidays
    or special hours.
    
    Parameters
    ----------
    start : pd.Timestamp, optional
        Start date for the calendar.
    end : pd.Timestamp, optional
        End date for the calendar.
    
    Attributes
    ----------
    name : str
        Calendar name ('24/7').
    tz : tzinfo
        Timezone (UTC for crypto markets).
    open_time : time
        Market open time (00:00:00 UTC).
    close_time : time
        Market close time (23:59:59 UTC).
    
    Examples
    --------
    >>> calendar = CryptoCalendar()
    >>> # Check if a date is a trading day (always True for crypto)
    >>> calendar.is_session(pd.Timestamp('2024-01-01', tz='UTC'))
    True
    """
    
    name = '24/7'
    tz = timezone('UTC')
    
    # Markets are always open
    open_time = time(0, 0, 0)
    close_time = time(23, 59, 59)
    
    @property
    def regular_holidays(self):
        """Return regular holidays (none for crypto)."""
        return HolidayCalendar([])
    
    @property
    def adhoc_holidays(self):
        """Return ad-hoc holidays (none for crypto)."""
        return []
    
    @property
    def special_opens(self):
        """Return special open times (none for crypto)."""
        return []
    
    @property
    def special_closes(self):
        """Return special close times (none for crypto)."""
        return []
    
    def is_session(self, dt):
        """
        Check if given datetime is a valid trading session.
        
        For crypto, every day is a trading session.
        
        Parameters
        ----------
        dt : pd.Timestamp
            Date to check.
            
        Returns
        -------
        bool
            Always True for crypto calendar.
        """
        return True
    
    def is_open_on_minute(self, dt):
        """
        Check if market is open at given minute.
        
        For crypto, market is always open.
        
        Parameters
        ----------
        dt : pd.Timestamp
            Datetime to check.
            
        Returns
        -------
        bool
            Always True for crypto calendar.
        """
        return True


class CryptoExchangeCalendar(CryptoCalendar):
    """
    Calendar for specific cryptocurrency exchanges.
    
    Some exchanges may have maintenance windows. This class allows
    defining exchange-specific downtimes.
    
    Parameters
    ----------
    exchange_name : str
        Name of the exchange.
    maintenance_windows : list of tuples, optional
        List of (start_time, end_time) tuples for regular maintenance.
    
    Examples
    --------
    >>> # Create calendar with weekly maintenance
    >>> calendar = CryptoExchangeCalendar(
    ...     exchange_name='binance',
    ...     maintenance_windows=[(time(2, 0), time(2, 30))]
    ... )
    """
    
    def __init__(self, exchange_name: str = 'crypto', 
                 maintenance_windows: list = None):
        super().__init__()
        self.exchange_name = exchange_name
        self.name = f'{exchange_name.upper()}_24/7'
        self.maintenance_windows = maintenance_windows or []
    
    def is_open_on_minute(self, dt):
        """
        Check if market is open at given minute.
        
        Accounts for exchange maintenance windows.
        
        Parameters
        ----------
        dt : pd.Timestamp
            Datetime to check.
            
        Returns
        -------
        bool
            True if market is open, False if in maintenance.
        """
        if not self.maintenance_windows:
            return True
        
        current_time = dt.time()
        
        for start_time, end_time in self.maintenance_windows:
            if start_time <= current_time <= end_time:
                return False
        
        return True


# Predefined exchange calendars
BINANCE_CALENDAR = CryptoExchangeCalendar('binance')
COINBASE_CALENDAR = CryptoExchangeCalendar('coinbase')
KRAKEN_CALENDAR = CryptoExchangeCalendar('kraken')
BITFINEX_CALENDAR = CryptoExchangeCalendar('bitfinex')


def get_crypto_calendar(exchange: str = None):
    """
    Get a crypto trading calendar.
    
    Parameters
    ----------
    exchange : str, optional
        Exchange name. If None, returns generic 24/7 calendar.
        
    Returns
    -------
    CryptoCalendar
        Trading calendar for the exchange.
        
    Examples
    --------
    >>> calendar = get_crypto_calendar('binance')
    >>> calendar = get_crypto_calendar()  # Generic 24/7 calendar
    """
    if exchange is None:
        return CryptoCalendar()
    
    exchange = exchange.lower()
    
    if exchange == 'binance':
        return BINANCE_CALENDAR
    elif exchange == 'coinbase':
        return COINBASE_CALENDAR
    elif exchange == 'kraken':
        return KRAKEN_CALENDAR
    elif exchange == 'bitfinex':
        return BITFINEX_CALENDAR
    else:
        return CryptoExchangeCalendar(exchange)

"""
Cryptocurrency asset classes.

This module defines asset types specifically for cryptocurrency trading.
"""
from typing import Optional
import pandas as pd


class CryptoAsset:
    """
    Cryptocurrency asset class.
    
    Represents a cryptocurrency asset with properties specific to
    crypto markets (24/7 trading, decimal precision, etc.).
    
    Parameters
    ----------
    symbol : str
        Cryptocurrency symbol (e.g., 'BTC', 'ETH').
    name : str
        Full name of the cryptocurrency.
    decimals : int, optional
        Number of decimal places for precision. Default is 8.
    min_trade_size : float, optional
        Minimum trade size. Default is 0.00000001.
    
    Attributes
    ----------
    symbol : str
        The cryptocurrency symbol.
    name : str
        Full name.
    decimals : int
        Decimal precision.
    min_trade_size : float
        Minimum trade size.
    
    Examples
    --------
    >>> btc = CryptoAsset(symbol='BTC', name='Bitcoin', decimals=8)
    >>> print(btc.symbol)
    BTC
    """
    
    def __init__(self, symbol: str, name: str, 
                 decimals: int = 8, min_trade_size: float = 0.00000001):
        self.symbol = symbol
        self.name = name
        self.decimals = decimals
        self.min_trade_size = min_trade_size
        self.asset_type = 'crypto'
    
    def __repr__(self):
        return f"CryptoAsset('{self.symbol}')"
    
    def __str__(self):
        return self.symbol
    
    def __eq__(self, other):
        if not isinstance(other, CryptoAsset):
            return False
        return self.symbol == other.symbol
    
    def __hash__(self):
        return hash(self.symbol)
    
    def round_trade_size(self, size: float) -> float:
        """
        Round trade size to valid precision.
        
        Parameters
        ----------
        size : float
            Desired trade size.
            
        Returns
        -------
        float
            Rounded trade size.
        """
        return round(size, self.decimals)
    
    def is_valid_trade_size(self, size: float) -> bool:
        """
        Check if trade size is valid.
        
        Parameters
        ----------
        size : float
            Trade size to validate.
            
        Returns
        -------
        bool
            True if valid, False otherwise.
        """
        return abs(size) >= self.min_trade_size


class CryptoPair:
    """
    Cryptocurrency trading pair.
    
    Represents a trading pair (e.g., BTC/USDT) with both base and quote assets.
    
    Parameters
    ----------
    base : CryptoAsset or str
        Base cryptocurrency.
    quote : CryptoAsset or str
        Quote currency.
    exchange : str, optional
        Exchange where the pair trades.
    
    Attributes
    ----------
    base : CryptoAsset
        Base cryptocurrency.
    quote : CryptoAsset
        Quote currency.
    exchange : str
        Exchange name.
    
    Examples
    --------
    >>> btc_usdt = CryptoPair('BTC', 'USDT', exchange='binance')
    >>> print(btc_usdt.symbol)
    BTC/USDT
    """
    
    def __init__(self, base, quote, exchange: Optional[str] = None):
        # Convert strings to CryptoAsset if needed
        if isinstance(base, str):
            self.base = CryptoAsset(symbol=base, name=base)
        else:
            self.base = base
        
        if isinstance(quote, str):
            self.quote = CryptoAsset(symbol=quote, name=quote)
        else:
            self.quote = quote
        
        self.exchange = exchange
        self.asset_type = 'crypto_pair'
    
    @property
    def symbol(self) -> str:
        """Get trading pair symbol."""
        return f"{self.base.symbol}/{self.quote.symbol}"
    
    @property
    def name(self) -> str:
        """Get full name of trading pair."""
        return f"{self.base.name}/{self.quote.name}"
    
    def __repr__(self):
        return f"CryptoPair('{self.symbol}')"
    
    def __str__(self):
        return self.symbol
    
    def __eq__(self, other):
        if not isinstance(other, CryptoPair):
            return False
        return (self.base == other.base and 
                self.quote == other.quote and
                self.exchange == other.exchange)
    
    def __hash__(self):
        return hash((self.base.symbol, self.quote.symbol, self.exchange))
    
    def to_exchange_format(self, format: str = 'ccxt') -> str:
        """
        Convert pair to exchange-specific format.
        
        Parameters
        ----------
        format : str, optional
            Exchange format ('ccxt', 'binance', etc.). Default is 'ccxt'.
            
        Returns
        -------
        str
            Formatted pair symbol.
        """
        if format == 'ccxt':
            return self.symbol
        elif format == 'binance':
            return f"{self.base.symbol}{self.quote.symbol}"
        elif format == 'coinbase':
            return f"{self.base.symbol}-{self.quote.symbol}"
        else:
            return self.symbol


# Predefined major cryptocurrencies
BTC = CryptoAsset('BTC', 'Bitcoin', decimals=8)
ETH = CryptoAsset('ETH', 'Ethereum', decimals=18)
USDT = CryptoAsset('USDT', 'Tether', decimals=6)
USDC = CryptoAsset('USDC', 'USD Coin', decimals=6)
BNB = CryptoAsset('BNB', 'Binance Coin', decimals=8)
XRP = CryptoAsset('XRP', 'Ripple', decimals=6)
ADA = CryptoAsset('ADA', 'Cardano', decimals=6)
SOL = CryptoAsset('SOL', 'Solana', decimals=9)
DOT = CryptoAsset('DOT', 'Polkadot', decimals=10)
DOGE = CryptoAsset('DOGE', 'Dogecoin', decimals=8)

# Common trading pairs
BTC_USDT = CryptoPair(BTC, USDT)
ETH_USDT = CryptoPair(ETH, USDT)
BTC_USDC = CryptoPair(BTC, USDC)
ETH_USDC = CryptoPair(ETH, USDC)
ETH_BTC = CryptoPair(ETH, BTC)

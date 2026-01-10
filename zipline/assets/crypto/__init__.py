"""
Cryptocurrency asset support for Zipline.

This module provides asset classes and utilities for trading
cryptocurrencies.
"""
from .asset import CryptoAsset, CryptoPair

__all__ = [
    'CryptoAsset',
    'CryptoPair',
]

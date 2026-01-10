"""
DEX Integration Module

Provides integration with decentralized exchanges.
"""

from zipline.blockchain.dex.aggregator import DEXAggregator
from zipline.blockchain.dex.uniswap import UniswapV3

__all__ = [
    'DEXAggregator',
    'UniswapV3',
]

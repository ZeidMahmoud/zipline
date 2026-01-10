"""
Blockchain and DeFi Integration Module

This module provides comprehensive blockchain and decentralized finance (DeFi)
integration capabilities for Zipline, enabling on-chain trading strategies,
wallet management, DEX interactions, and DeFi protocol integrations.
"""

from zipline.blockchain.wallet.manager import WalletManager
from zipline.blockchain.dex.aggregator import DEXAggregator

__all__ = [
    'WalletManager',
    'DEXAggregator',
]

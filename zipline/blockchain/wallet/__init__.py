"""
Multi-chain Wallet Management

Provides secure wallet management across multiple blockchain networks including
Ethereum, Solana, and Bitcoin.
"""

from zipline.blockchain.wallet.manager import WalletManager
from zipline.blockchain.wallet.ethereum import EthereumWallet
from zipline.blockchain.wallet.solana import SolanaWallet
from zipline.blockchain.wallet.bitcoin import BitcoinWallet

__all__ = [
    'WalletManager',
    'EthereumWallet',
    'SolanaWallet',
    'BitcoinWallet',
]

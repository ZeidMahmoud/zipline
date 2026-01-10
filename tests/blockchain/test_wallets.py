"""
Tests for blockchain wallet management
"""

import unittest
from zipline.blockchain.wallet.manager import WalletManager, WalletType


class TestWalletManager(unittest.TestCase):
    """Test wallet manager functionality"""
    
    def setUp(self):
        """Set up test wallet manager"""
        self.manager = WalletManager()
    
    def test_create_ethereum_wallet(self):
        """Test creating Ethereum wallet"""
        wallet = self.manager.create_wallet(
            WalletType.ETHEREUM,
            name="test_eth_wallet"
        )
        
        self.assertIsNotNone(wallet)
        self.assertIsNotNone(wallet.get_address())
        self.assertTrue(wallet.get_address().startswith("0x"))
    
    def test_create_solana_wallet(self):
        """Test creating Solana wallet"""
        wallet = self.manager.create_wallet(
            WalletType.SOLANA,
            name="test_sol_wallet"
        )
        
        self.assertIsNotNone(wallet)
        self.assertIsNotNone(wallet.get_address())
    
    def test_list_wallets(self):
        """Test listing all wallets"""
        self.manager.create_wallet(WalletType.ETHEREUM, "wallet1")
        self.manager.create_wallet(WalletType.SOLANA, "wallet2")
        
        wallets = self.manager.list_wallets()
        self.assertEqual(len(wallets), 2)
        self.assertIn("wallet1", wallets)
        self.assertIn("wallet2", wallets)


if __name__ == '__main__':
    unittest.main()

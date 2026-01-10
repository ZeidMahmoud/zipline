Blockchain & DeFi Integration
==============================

Zipline now includes comprehensive blockchain and DeFi integration capabilities,
enabling you to trade on decentralized exchanges, interact with DeFi protocols,
and manage multi-chain wallets.

Installation
------------

Install blockchain support with::

    pip install zipline[blockchain]

For full DeFi features::

    pip install zipline[defi]

For everything::

    pip install zipline[full_ecosystem]

Wallet Management
-----------------

The wallet management system supports multiple blockchains:

.. code-block:: python

    from zipline.blockchain.wallet.manager import WalletManager, WalletType
    
    # Initialize wallet manager
    manager = WalletManager()
    
    # Create Ethereum wallet
    eth_wallet = manager.create_wallet(
        WalletType.ETHEREUM,
        name="my_eth_wallet"
    )
    
    # Get wallet address
    address = eth_wallet.get_address()
    print(f"Wallet address: {address}")
    
    # Check balance
    balance = eth_wallet.get_balance()
    print(f"Balance: {balance} ETH")

Supported Blockchains
~~~~~~~~~~~~~~~~~~~~~

- **Ethereum**: ETH and ERC-20 tokens with EIP-1559 support
- **Solana**: SOL and SPL tokens with fast transaction signing
- **Bitcoin**: BTC with SegWit and UTXO management

DEX Trading
-----------

Trade on decentralized exchanges with built-in aggregation:

.. code-block:: python

    from zipline.blockchain.dex.aggregator import DEXAggregator
    
    # Initialize DEX aggregator
    aggregator = DEXAggregator(
        wallet_address="0x...",
        enable_mev_protection=True  # Flashbots protection
    )
    
    # Get best quote across all DEXs
    quote = aggregator.get_best_quote(
        token_in="USDC",
        token_out="ETH",
        amount_in=1000
    )
    
    # Execute swap
    tx_hash = aggregator.execute_swap(quote, slippage=0.5)

Supported DEXs
~~~~~~~~~~~~~~

- Uniswap V3
- SushiSwap
- Curve Finance
- PancakeSwap (BSC)
- Jupiter (Solana)
- And more via aggregators (1inch, Paraswap, 0x)

DeFi Protocols
--------------

Interact with major DeFi lending and yield protocols:

.. code-block:: python

    from zipline.blockchain.defi.lending import AaveV3
    
    # Initialize Aave V3
    aave = AaveV3(wallet_address="0x...")
    
    # Deposit USDC to earn interest
    aave.deposit("USDC", amount=1000)
    
    # Borrow ETH against your collateral
    aave.borrow("ETH", amount=0.5)
    
    # Monitor health factor
    health = aave.get_health_factor()
    if health < 1.2:
        print("⚠️ Low health factor - add collateral!")

Supported Protocols
~~~~~~~~~~~~~~~~~~~

**Lending:**
- Aave V3
- Compound V3
- MakerDAO

**Yield Farming:**
- Yearn Finance
- Convex Finance

**Derivatives:**
- GMX
- dYdX
- Synthetix

Security Considerations
-----------------------

When working with blockchain:

1. **Always test on testnet first**
2. **Never commit private keys to source control**
3. **Use hardware wallets for production**
4. **Enable MEV protection for large trades**
5. **Monitor gas prices and set appropriate limits**
6. **Keep small amounts in hot wallets**

Example: DEX Arbitrage Bot
---------------------------

See ``examples/blockchain/dex_arbitrage.py`` for a complete example of building
a DEX arbitrage bot.

Further Reading
---------------

- :doc:`/blockchain/wallets` - Detailed wallet management
- :doc:`/blockchain/dex` - DEX integration guide
- :doc:`/blockchain/defi` - DeFi protocols
- :doc:`/blockchain/strategies` - Web3 trading strategies

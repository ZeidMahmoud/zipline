"""
DEX Arbitrage Bot Example

This example demonstrates how to build a simple DEX arbitrage bot
that finds price differences across decentralized exchanges.
"""

from zipline.blockchain.wallet.manager import WalletManager, WalletType
from zipline.blockchain.dex.aggregator import DEXAggregator, DEXProtocol
from decimal import Decimal
import time


def main():
    """
    Simple DEX arbitrage strategy
    
    This bot:
    1. Monitors prices across multiple DEXs
    2. Identifies arbitrage opportunities
    3. Executes trades when profitable
    """
    
    # Initialize wallet
    print("Initializing wallet...")
    wallet_manager = WalletManager()
    
    # For demo purposes, we'll use a test wallet
    # In production, use secure key management
    wallet = wallet_manager.create_wallet(
        WalletType.ETHEREUM,
        name="arbitrage_wallet",
        # mnemonic="your twelve word mnemonic phrase here"
    )
    
    print(f"Wallet address: {wallet.get_address()}")
    
    # Initialize DEX aggregator
    print("Initializing DEX aggregator...")
    aggregator = DEXAggregator(
        wallet_address=wallet.get_address(),
        enable_mev_protection=True,  # Protect against front-running
        testnet=True  # Use testnet for demo
    )
    
    # Tokens to monitor
    USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    WETH = "0xC02aaA39b223FE8D0A3e5C4F27eAD9083C756Cc2"
    
    # Arbitrage parameters
    MIN_PROFIT_USD = 50  # Minimum profit threshold
    SLIPPAGE = 0.5  # 0.5% slippage tolerance
    
    print("Starting arbitrage monitoring...")
    print(f"Min profit threshold: ${MIN_PROFIT_USD}")
    
    while True:
        try:
            # Get quotes from all DEXs
            amount_in = 10000 * 10**6  # 10,000 USDC (6 decimals)
            
            print("\nComparing prices across DEXs...")
            quotes = aggregator.compare_prices(
                token_in=USDC,
                token_out=WETH,
                amount_in=amount_in
            )
            
            # Display quotes
            for i, quote in enumerate(quotes[:3], 1):
                print(f"{i}. {quote.get('protocol', 'Unknown')}: {quote.get('amount_out', 0)} WETH")
            
            # Check for arbitrage opportunity
            if len(quotes) >= 2:
                best_buy = quotes[0]  # Best price to buy
                best_sell = quotes[-1]  # Worst price (where we could sell back)
                
                # Calculate potential profit
                # This is simplified - real arb would be more complex
                price_diff = best_buy['amount_out'] - best_sell['amount_out']
                
                # Estimate profit in USD (simplified)
                # In reality, you'd need to account for gas costs
                estimated_profit = price_diff * 2000  # Assuming ETH = $2000
                
                if estimated_profit > MIN_PROFIT_USD:
                    print(f"\n💰 Arbitrage opportunity found!")
                    print(f"Potential profit: ${estimated_profit:.2f}")
                    print(f"Buy on: {best_buy.get('protocol')}")
                    print(f"Sell on: {best_sell.get('protocol')}")
                    
                    # In production, execute the arbitrage here
                    # For demo, we just log the opportunity
                    print("⚠️  Demo mode - not executing trade")
                    
                    # Execute trade (commented out for demo)
                    # tx_hash = aggregator.execute_swap(
                    #     best_buy,
                    #     slippage=SLIPPAGE
                    # )
                    # print(f"Transaction: {tx_hash}")
                
                else:
                    print(f"No profitable arbitrage. Diff: ${estimated_profit:.2f}")
            
            # Wait before next check
            time.sleep(10)  # Check every 10 seconds
            
        except KeyboardInterrupt:
            print("\n\nStopping arbitrage bot...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)  # Wait longer on error


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════╗
    ║   Zipline DEX Arbitrage Bot Example   ║
    ╚════════════════════════════════════════╝
    
    ⚠️  This is a demo example for educational purposes.
    ⚠️  Requires: pip install zipline[blockchain]
    ⚠️  Use testnet for learning and testing.
    
    Press Ctrl+C to stop.
    """)
    
    main()

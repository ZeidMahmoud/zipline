#!/usr/bin/env python
"""
Live Trading with Alpaca Example.

This example demonstrates how to use Zipline for live trading
with the Alpaca broker.
"""
import os
from datetime import datetime

try:
    from zipline.live import AlpacaBroker, LiveExecutionEngine
    from zipline.live.data_feed import AlpacaDataFeed
except ImportError:
    print("Live trading modules not installed")
    print("Install with: pip install 'zipline[live]'")
    exit(1)


def setup_live_trading():
    """
    Setup live trading with Alpaca.
    
    This function demonstrates how to configure live trading.
    """
    # Get API credentials from environment
    api_key = os.environ.get('ALPACA_API_KEY')
    api_secret = os.environ.get('ALPACA_API_SECRET')
    
    if not api_key or not api_secret:
        print("Error: Alpaca API credentials not found!")
        print("Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        return None, None
    
    # Use paper trading URL by default
    base_url = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    print(f"Connecting to Alpaca ({base_url})...")
    
    # Create broker
    broker = AlpacaBroker(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url
    )
    
    # Connect to broker
    if not broker.connect():
        print("Failed to connect to Alpaca")
        return None, None
    
    print("Connected to Alpaca successfully!")
    
    # Create execution engine
    engine = LiveExecutionEngine(broker)
    engine.connect()
    
    # Create data feed
    data_feed = AlpacaDataFeed(api_key, api_secret)
    data_feed.connect()
    
    return engine, data_feed


def simple_trading_strategy(engine, data_feed):
    """
    Simple live trading strategy.
    
    This is a basic example of a live trading strategy.
    """
    print("\nRunning simple trading strategy...")
    
    # Get account balance
    balance = engine.get_account_balance()
    print(f"\nAccount Balance:")
    print(f"  Cash: ${balance['cash']:,.2f}")
    print(f"  Portfolio Value: ${balance['portfolio_value']:,.2f}")
    print(f"  Buying Power: ${balance['buying_power']:,.2f}")
    
    # Get current positions
    positions = engine.get_positions()
    print(f"\nCurrent Positions: {len(positions)}")
    for symbol, position in positions.items():
        print(f"  {symbol}: {position.amount} shares @ ${position.last_sale_price:.2f}")
    
    # Example: Submit a test order (commented out for safety)
    # from zipline.assets import Equity
    # asset = Equity(symbol='AAPL')
    # order = engine.submit_market_order(asset, 1)  # Buy 1 share of AAPL
    # print(f"\nSubmitted order: {order.id}")
    
    print("\n" + "=" * 50)
    print("Live trading setup complete!")
    print("=" * 50)
    print("\nTo place orders, uncomment the order submission code.")
    print("⚠️  Remember: This uses real money (or paper trading account)!")


def order_callback(order):
    """
    Callback function for order events.
    
    This function is called when order status changes.
    """
    print(f"Order event: {order.id} - Status: {order.status}")
    if order.status == 'filled':
        print(f"  Filled {order.filled} shares at ${order.filled * order.limit_price:.2f}")


def main():
    """Main function."""
    print("=" * 50)
    print("Zipline Live Trading with Alpaca")
    print("=" * 50)
    print()
    
    # Setup live trading
    engine, data_feed = setup_live_trading()
    
    if engine is None:
        return
    
    # Register order callbacks
    engine.register_callback('on_order_filled', order_callback)
    
    # Run strategy
    try:
        simple_trading_strategy(engine, data_feed)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        # Cleanup
        engine.disconnect()
        if data_feed:
            data_feed.disconnect()
        print("Disconnected from Alpaca")


if __name__ == '__main__':
    main()

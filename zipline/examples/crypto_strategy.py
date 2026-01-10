#!/usr/bin/env python
"""
Cryptocurrency Trading Strategy Example.

This example demonstrates how to trade cryptocurrencies using Zipline
with the crypto data bundle and 24/7 calendar.
"""
from zipline.api import (
    order_target_percent,
    schedule_function,
    symbol,
)

try:
    from zipline.assets.crypto import BTC_USDT, ETH_USDT, CryptoPair
    from zipline.utils.calendars.crypto_calendar import get_crypto_calendar
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Required packages not installed: {e}")
    print("Install with: pip install 'zipline[crypto]'")
    exit(1)


def initialize(context):
    """
    Initialize crypto trading strategy.
    
    This function is called once at the start of the simulation.
    """
    # Crypto pairs to trade
    context.pairs = [
        symbol('BTCUSDT'),  # BTC/USDT
        symbol('ETHUSDT'),  # ETH/USDT
        symbol('BNBUSDT'),  # BNB/USDT
    ]
    
    # Trading parameters
    context.lookback = 24  # 24 hours for crypto
    context.rebalance_hours = 4  # Rebalance every 4 hours
    
    # Schedule rebalance (crypto trades 24/7)
    # In production, would use schedule_function with crypto calendar
    
    print("Cryptocurrency trading strategy initialized")
    print(f"Trading pairs: {[str(p) for p in context.pairs]}")


def handle_data(context, data):
    """
    Called every bar (e.g., every hour for hourly data).
    
    This is the main trading logic for crypto.
    """
    # Get current hour
    current_hour = data.current_dt.hour
    
    # Only rebalance every N hours
    if current_hour % context.rebalance_hours != 0:
        return
    
    # Calculate momentum for each pair
    prices = {}
    momentum = {}
    
    for pair in context.pairs:
        try:
            # Get recent price history
            price_history = data.history(pair, 'close', context.lookback, '1h')
            
            if len(price_history) < context.lookback:
                continue
            
            prices[pair] = price_history
            
            # Calculate momentum (change over lookback period)
            momentum[pair] = (price_history.iloc[-1] - price_history.iloc[0]) / price_history.iloc[0]
            
        except Exception as e:
            print(f"Error processing {pair}: {e}")
            continue
    
    if not momentum:
        return
    
    # Rank pairs by momentum
    ranked = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
    
    # Long top performers, short bottom performers
    n_long = len(ranked) // 2
    long_pairs = [pair for pair, _ in ranked[:n_long]]
    short_pairs = [pair for pair, _ in ranked[n_long:]]
    
    # Calculate position sizes
    target_weight = 0.9 / max(len(long_pairs), 1)
    short_weight = -0.5 / max(len(short_pairs), 1)
    
    # Rebalance portfolio
    for pair in context.pairs:
        if pair in long_pairs:
            order_target_percent(pair, target_weight)
        elif pair in short_pairs:
            order_target_percent(pair, short_weight)
        else:
            order_target_percent(pair, 0.0)
    
    print(f"{data.current_dt}: Rebalanced - Long: {[str(p) for p in long_pairs]}")


def analyze(context, perf):
    """
    Analyze results after backtest.
    
    This function is called after the backtest completes.
    """
    import matplotlib.pyplot as plt
    
    # Plot performance
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Portfolio value
    perf['portfolio_value'].plot(ax=axes[0])
    axes[0].set_ylabel('Portfolio Value (USDT)')
    axes[0].set_title('Crypto Trading Strategy Performance')
    axes[0].grid(True)
    
    # Returns distribution
    perf['returns'].hist(bins=50, ax=axes[1])
    axes[1].set_xlabel('Returns')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Returns Distribution')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/tmp/crypto_strategy_results.png')
    print("Results saved to /tmp/crypto_strategy_results.png")
    
    # Calculate statistics
    total_return = (perf['portfolio_value'].iloc[-1] - 
                   perf['portfolio_value'].iloc[0]) / perf['portfolio_value'].iloc[0]
    
    # Crypto trades 24/7, so use 365*24 for hourly data
    returns_per_year = 365 * 24
    sharpe = perf['returns'].mean() / perf['returns'].std() * np.sqrt(returns_per_year)
    
    print(f"\nCrypto Strategy Statistics:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {perf['max_drawdown'].min():.2%}")
    
    # Crypto-specific metrics
    volatility = perf['returns'].std() * np.sqrt(returns_per_year)
    print(f"Annualized Volatility: {volatility:.2%}")
    
    # Win rate
    winning_days = (perf['returns'] > 0).sum()
    total_days = len(perf['returns'])
    win_rate = winning_days / total_days
    print(f"Win Rate: {win_rate:.2%}")


def _test_args():
    """Extra arguments for automated testing."""
    import pandas as pd
    return {
        'start': pd.Timestamp('2023-01-01', tz='utc'),
        'end': pd.Timestamp('2023-06-01', tz='utc'),
        'data_frequency': 'daily',  # or 'hourly' for intraday
    }


if __name__ == '__main__':
    print("Cryptocurrency Trading Strategy Example")
    print("=" * 50)
    print("\nThis example demonstrates crypto trading with Zipline.")
    print("\nFeatures:")
    print("  - 24/7 trading calendar")
    print("  - Momentum-based strategy")
    print("  - Multiple crypto pairs")
    print("\nTo run this strategy:")
    print("  1. Ingest crypto data:")
    print("     zipline ingest -b crypto")
    print("  2. Run backtest:")
    print("     zipline run -f crypto_strategy.py --start 2023-1-1 --end 2023-6-1 -b crypto")
    print("\nNote: Requires crypto bundle.")
    print("  Set CRYPTO_EXCHANGE and CRYPTO_PAIRS environment variables.")

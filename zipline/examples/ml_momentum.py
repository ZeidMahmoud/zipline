#!/usr/bin/env python
"""
ML-Enhanced Momentum Strategy Example.

This example demonstrates how to use machine learning with Zipline
to enhance a traditional momentum trading strategy.
"""
from zipline.api import (
    order_target_percent,
    schedule_function,
    date_rules,
    time_rules,
    set_slippage,
    set_commission,
    symbol,
)
from zipline.finance import commission, slippage

try:
    from zipline.ml import MLPredictionFactor, SklearnModelWrapper, TechnicalFeatures
    from zipline.pipeline import Pipeline
    from zipline.pipeline.data import USEquityPricing
    from zipline.pipeline.factors import SimpleMovingAverage, Returns
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Required packages not installed: {e}")
    print("Install with: pip install 'zipline[ml]'")
    exit(1)


def initialize(context):
    """
    Initialize algorithm.
    
    This function is called once at the start of the simulation.
    """
    # Set commission and slippage
    context.set_commission(commission.PerShare(cost=0.001, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())
    
    # Universe of stocks to trade
    context.stocks = [symbol('AAPL'), symbol('MSFT'), symbol('GOOGL'), 
                     symbol('AMZN'), symbol('FB')]
    
    # ML model for predictions (in practice, would train offline)
    context.model = None
    
    # Schedule rebalance function
    schedule_function(
        rebalance,
        date_rules.week_start(),
        time_rules.market_open(hours=1)
    )
    
    print("ML Momentum Strategy initialized")


def before_trading_start(context, data):
    """
    Called every day before market opens.
    
    This is where we can prepare for the day's trading.
    """
    pass


def rebalance(context, data):
    """
    Rebalance portfolio based on ML predictions.
    
    This function is called on schedule to rebalance positions.
    """
    # Calculate momentum scores
    prices = data.history(context.stocks, 'close', 20, '1d')
    
    if prices.empty or len(prices) < 20:
        return
    
    scores = {}
    
    for stock in context.stocks:
        if stock not in prices.columns:
            continue
        
        stock_prices = prices[stock].values
        
        # Skip if insufficient data
        if len(stock_prices) < 20 or np.isnan(stock_prices).any():
            continue
        
        # Calculate features
        returns = np.diff(stock_prices) / stock_prices[:-1]
        momentum = (stock_prices[-1] - stock_prices[0]) / stock_prices[0]
        volatility = np.std(returns)
        
        # Simple momentum score (in practice, use ML model)
        score = momentum / volatility if volatility > 0 else 0
        scores[stock] = score
    
    if not scores:
        return
    
    # Rank stocks by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Long top 3, short bottom 2
    long_stocks = [stock for stock, _ in ranked[:3]]
    short_stocks = [stock for stock, _ in ranked[-2:]]
    
    # Set target positions
    target_weight = 0.3
    
    for stock in context.stocks:
        if stock in long_stocks:
            order_target_percent(stock, target_weight)
        elif stock in short_stocks:
            order_target_percent(stock, -0.2)
        else:
            order_target_percent(stock, 0.0)
    
    print(f"Rebalanced: Long {[str(s) for s in long_stocks]}, "
          f"Short {[str(s) for s in short_stocks]}")


def handle_data(context, data):
    """
    Called every market minute.
    
    This is the main trading logic. For this strategy, we only
    trade on rebalance, so this function is mostly empty.
    """
    pass


def analyze(context, perf):
    """
    Analyze results after backtest.
    
    This function is called after the backtest completes.
    """
    import matplotlib.pyplot as plt
    
    # Plot performance
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Portfolio value
    perf['portfolio_value'].plot(ax=ax1)
    ax1.set_ylabel('Portfolio Value')
    ax1.set_title('ML Momentum Strategy Performance')
    ax1.grid(True)
    
    # Returns
    perf['returns'].plot(ax=ax2)
    ax2.set_ylabel('Returns')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/tmp/ml_momentum_results.png')
    print("Results saved to /tmp/ml_momentum_results.png")
    
    # Print statistics
    total_return = (perf['portfolio_value'].iloc[-1] - 
                   perf['portfolio_value'].iloc[0]) / perf['portfolio_value'].iloc[0]
    sharpe = perf['returns'].mean() / perf['returns'].std() * np.sqrt(252)
    
    print(f"\nStrategy Statistics:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {(perf['max_drawdown'].min()):.2%}")


def _test_args():
    """Extra arguments for automated testing."""
    import pandas as pd
    return {
        'start': pd.Timestamp('2018-01-01', tz='utc'),
        'end': pd.Timestamp('2019-01-01', tz='utc'),
    }


if __name__ == '__main__':
    print("ML Momentum Strategy Example")
    print("=" * 50)
    print("\nThis example demonstrates ML-enhanced momentum trading.")
    print("\nTo run this strategy:")
    print("  zipline run -f ml_momentum.py --start 2018-1-1 --end 2019-1-1")
    print("\nNote: Requires ML dependencies.")
    print("  pip install 'zipline[ml]'")

"""
Cryptocurrency data bundle for Zipline.

This module provides integration with cryptocurrency exchanges
for historical OHLCV data on crypto trading pairs.
"""
import pandas as pd
from logbook import Logger

from . import core as bundles

log = Logger(__name__)


@bundles.register('crypto', calendar_name='24/7')
def crypto_bundle(environ,
                 asset_db_writer,
                 minute_bar_writer,
                 daily_bar_writer,
                 adjustment_writer,
                 calendar,
                 start_session,
                 end_session,
                 cache,
                 show_progress,
                 output_dir):
    """
    Cryptocurrency data bundle.
    
    Downloads historical cryptocurrency data from exchanges via CCXT.
    Requires CRYPTO_EXCHANGE (default: 'binance') and optionally
    CRYPTO_PAIRS (default: 'BTC/USDT,ETH/USDT') environment variables.
    
    Parameters
    ----------
    environ : dict
        Environment variables, should contain 'CRYPTO_EXCHANGE'
        and optionally 'CRYPTO_PAIRS'.
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError(
            "ccxt package required for crypto bundle. "
            "Install it with: pip install ccxt"
        )
    
    exchange_name = environ.get('CRYPTO_EXCHANGE', 'binance')
    pairs_str = environ.get('CRYPTO_PAIRS', 'BTC/USDT,ETH/USDT,BNB/USDT')
    pairs = [p.strip() for p in pairs_str.split(',')]
    
    log.info(f"Crypto bundle: downloading {len(pairs)} pairs from {exchange_name}")
    
    # Initialize exchange
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            'enableRateLimit': True,
        })
    except AttributeError:
        raise ValueError(f"Unknown exchange: {exchange_name}")
    
    # Prepare asset metadata for crypto assets
    asset_metadata = []
    for idx, pair in enumerate(pairs):
        # Convert pair format (e.g., BTC/USDT -> BTCUSDT)
        symbol = pair.replace('/', '')
        
        asset_metadata.append({
            'symbol': symbol,
            'asset_name': pair,
            'start_date': start_session,
            'end_date': end_session,
            'exchange': exchange_name.upper(),
            'auto_close_date': end_session + pd.Timedelta(days=1),
        })
    
    asset_metadata = pd.DataFrame(asset_metadata)
    asset_db_writer.write(equities=asset_metadata)
    
    # Download and write data for each pair
    def gen_daily_bars():
        for idx, pair in enumerate(pairs):
            try:
                log.info(f"Downloading {pair} from {exchange_name}")
                
                # Fetch OHLCV data
                since = int(start_session.timestamp() * 1000)
                ohlcv = exchange.fetch_ohlcv(
                    pair,
                    timeframe='1d',
                    since=since,
                    limit=1000
                )
                
                if not ohlcv:
                    log.warning(f"No data for {pair}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('date')
                df = df.drop('timestamp', axis=1)
                
                # Filter to date range
                df = df[(df.index >= start_session) & (df.index <= end_session)]
                df = df.sort_index()
                
                if df.empty:
                    log.warning(f"No data in range for {pair}")
                    continue
                
                yield idx, df[['open', 'high', 'low', 'close', 'volume']]
                
            except Exception as e:
                log.error(f"Failed to download {pair}: {e}")
                continue
    
    daily_bar_writer.write(gen_daily_bars(), show_progress=show_progress)
    log.info("Crypto bundle ingestion complete")

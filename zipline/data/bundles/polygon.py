"""
Polygon.io data bundle for Zipline.

This module provides integration with Polygon.io API for
high-quality market data including tick-level data.
"""
import pandas as pd
from logbook import Logger

from . import core as bundles

log = Logger(__name__)


@bundles.register('polygon', calendar_name='NYSE')
def polygon_bundle(environ,
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
    Polygon.io data bundle.
    
    Downloads historical stock data from Polygon.io API.
    Requires POLYGON_API_KEY environment variable.
    
    Parameters
    ----------
    environ : dict
        Environment variables, should contain 'POLYGON_API_KEY'
        and optionally 'POLYGON_SYMBOLS'.
    """
    api_key = environ.get('POLYGON_API_KEY')
    if not api_key:
        raise ValueError(
            "Polygon.io API key required. "
            "Set POLYGON_API_KEY environment variable."
        )
    
    symbols_str = environ.get('POLYGON_SYMBOLS', 'SPY,AAPL,MSFT')
    symbols = [s.strip() for s in symbols_str.split(',')]
    
    log.info(f"Polygon bundle: downloading {len(symbols)} symbols")
    
    try:
        import requests
    except ImportError:
        raise ImportError("requests package required for Polygon bundle")
    
    # Prepare asset metadata
    asset_metadata = []
    for symbol in symbols:
        asset_metadata.append({
            'symbol': symbol,
            'asset_name': symbol,
            'start_date': start_session,
            'end_date': end_session,
            'exchange': 'NYSE',
            'auto_close_date': end_session + pd.Timedelta(days=1),
        })
    
    asset_metadata = pd.DataFrame(asset_metadata)
    asset_db_writer.write(equities=asset_metadata)
    
    # Download and write data for each symbol
    def gen_daily_bars():
        for idx, symbol in enumerate(symbols):
            try:
                log.info(f"Downloading {symbol} from Polygon.io")
                
                # Format dates for API
                start_str = start_session.strftime('%Y-%m-%d')
                end_str = end_session.strftime('%Y-%m-%d')
                
                url = (
                    f"https://api.polygon.io/v2/aggs/ticker/{symbol}/"
                    f"range/1/day/{start_str}/{end_str}?"
                    f"adjusted=true&sort=asc&apiKey={api_key}"
                )
                
                response = requests.get(url)
                data = response.json()
                
                if data.get('status') != 'OK' or 'results' not in data:
                    log.warning(f"No data for {symbol}")
                    continue
                
                # Parse results
                results = data['results']
                df = pd.DataFrame(results)
                
                # Convert timestamp to datetime
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df = df.set_index('date')
                
                # Rename columns to match Zipline format
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume',
                })
                
                df = df.sort_index()
                
                if df.empty:
                    log.warning(f"No data for {symbol}")
                    continue
                
                yield idx, df[['open', 'high', 'low', 'close', 'volume']]
                
            except Exception as e:
                log.error(f"Failed to download {symbol}: {e}")
                continue
    
    daily_bar_writer.write(gen_daily_bars(), show_progress=show_progress)
    log.info("Polygon bundle ingestion complete")

"""
Alpha Vantage data bundle for Zipline.

This module provides integration with Alpha Vantage API for
historical stock data.
"""
import pandas as pd
from logbook import Logger

from . import core as bundles

log = Logger(__name__)


@bundles.register('alpha_vantage', calendar_name='NYSE')
def alpha_vantage_bundle(environ,
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
    Alpha Vantage data bundle.
    
    Downloads historical stock data from Alpha Vantage API.
    Requires ALPHAVANTAGE_API_KEY environment variable.
    
    Parameters
    ----------
    environ : dict
        Environment variables, should contain 'ALPHAVANTAGE_API_KEY'
        and optionally 'ALPHAVANTAGE_SYMBOLS'.
    """
    api_key = environ.get('ALPHAVANTAGE_API_KEY')
    if not api_key:
        raise ValueError(
            "Alpha Vantage API key required. "
            "Set ALPHAVANTAGE_API_KEY environment variable."
        )
    
    symbols_str = environ.get('ALPHAVANTAGE_SYMBOLS', 'SPY,AAPL,MSFT')
    symbols = [s.strip() for s in symbols_str.split(',')]
    
    log.info(f"Alpha Vantage bundle: downloading {len(symbols)} symbols")
    
    try:
        import requests
    except ImportError:
        raise ImportError("requests package required for Alpha Vantage bundle")
    
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
                log.info(f"Downloading {symbol} from Alpha Vantage")
                
                url = (
                    f"https://www.alphavantage.co/query?"
                    f"function=TIME_SERIES_DAILY_ADJUSTED&"
                    f"symbol={symbol}&"
                    f"outputsize=full&"
                    f"apikey={api_key}"
                )
                
                response = requests.get(url)
                data = response.json()
                
                if 'Time Series (Daily)' not in data:
                    log.warning(f"No data for {symbol}")
                    continue
                
                # Parse time series data
                ts_data = data['Time Series (Daily)']
                df = pd.DataFrame.from_dict(ts_data, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Rename columns
                df = df.rename(columns={
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close',
                    '5. adjusted close': 'adj_close',
                    '6. volume': 'volume',
                    '7. dividend amount': 'dividend',
                    '8. split coefficient': 'split',
                })
                
                # Convert to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Filter to date range
                df = df[(df.index >= start_session) & (df.index <= end_session)]
                
                if df.empty:
                    log.warning(f"No data in range for {symbol}")
                    continue
                
                yield idx, df[['open', 'high', 'low', 'close', 'volume']]
                
            except Exception as e:
                log.error(f"Failed to download {symbol}: {e}")
                continue
    
    daily_bar_writer.write(gen_daily_bars(), show_progress=show_progress)
    log.info("Alpha Vantage bundle ingestion complete")

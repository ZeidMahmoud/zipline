"""
Yahoo Finance data bundle for Zipline.

This module provides integration with Yahoo Finance for free historical
OHLCV data with dividend and split adjustments.
"""
import pandas as pd
import numpy as np
from logbook import Logger
from typing import Optional

from . import core as bundles

log = Logger(__name__)


def yahoo_equities(symbols, start=None, end=None):
    """
    Download equity data from Yahoo Finance.
    
    Parameters
    ----------
    symbols : list of str
        List of ticker symbols to download.
    start : datetime-like, optional
        Start date for data download.
    end : datetime-like, optional
        End date for data download.
        
    Returns
    -------
    DataFrame
        Multi-index DataFrame with (symbol, date) index.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for Yahoo Finance bundle. "
            "Install it with: pip install yfinance"
        )
    
    dfs = []
    
    for symbol in symbols:
        try:
            log.info(f"Downloading {symbol} from Yahoo Finance")
            ticker = yf.Ticker(symbol)
            
            # Download historical data
            df = ticker.history(start=start, end=end, auto_adjust=False)
            
            if df.empty:
                log.warning(f"No data found for {symbol}")
                continue
            
            # Rename columns to match Zipline format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'ex_dividend',
                'Stock Splits': 'split_ratio'
            })
            
            # Add symbol column
            df['symbol'] = symbol
            df.index.name = 'date'
            df = df.reset_index()
            
            dfs.append(df)
            
        except Exception as e:
            log.error(f"Failed to download {symbol}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No data downloaded for any symbols")
    
    # Combine all dataframes
    result = pd.concat(dfs, ignore_index=True)
    result = result.set_index(['symbol', 'date']).sort_index()
    
    return result


@bundles.register('yahoo', calendar_name='NYSE')
def yahoo_bundle(environ,
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
    Yahoo Finance data bundle.
    
    This bundle downloads free historical data from Yahoo Finance.
    Configure the symbols to download via the YAHOO_SYMBOLS environment variable.
    
    Parameters
    ----------
    environ : dict
        Environment variables, should contain 'YAHOO_SYMBOLS' with
        comma-separated list of symbols.
    asset_db_writer : AssetDBWriter
        Asset database writer.
    minute_bar_writer : BcolzMinuteBarWriter
        Minute bar writer (not used for daily data).
    daily_bar_writer : BcolzDailyBarWriter
        Daily bar writer.
    adjustment_writer : SQLiteAdjustmentWriter
        Adjustment writer for splits and dividends.
    calendar : TradingCalendar
        Trading calendar for the exchange.
    start_session : pd.Timestamp
        Start date for the bundle.
    end_session : pd.Timestamp
        End date for the bundle.
    cache : DataFrameCache
        Cache for storing intermediate results.
    show_progress : bool
        Whether to show progress bars.
    output_dir : str
        Output directory for bundle data.
    """
    # Get symbols from environment
    symbols_str = environ.get('YAHOO_SYMBOLS', 'SPY,AAPL,MSFT,GOOGL,AMZN')
    symbols = [s.strip() for s in symbols_str.split(',')]
    
    log.info(f"Yahoo bundle: downloading {len(symbols)} symbols")
    
    # Download data
    raw_data = yahoo_equities(symbols, start=start_session, end=end_session)
    
    # Prepare asset metadata
    asset_metadata = []
    for symbol in symbols:
        asset_metadata.append({
            'symbol': symbol,
            'asset_name': symbol,
            'start_date': start_session,
            'end_date': end_session,
            'exchange': 'NYSE',  # Default exchange
            'auto_close_date': end_session + pd.Timedelta(days=1),
        })
    
    asset_metadata = pd.DataFrame(asset_metadata)
    
    # Write asset metadata
    asset_db_writer.write(equities=asset_metadata)
    
    # Prepare daily bars
    daily_bars = []
    splits = []
    dividends = []
    
    for symbol in symbols:
        if symbol not in raw_data.index.get_level_values('symbol'):
            log.warning(f"No data for {symbol}, skipping")
            continue
        
        symbol_data = raw_data.loc[symbol]
        
        # Get asset ID from the asset writer
        # Note: In actual implementation, would need proper asset lookup
        sid = asset_metadata[asset_metadata['symbol'] == symbol].index[0]
        
        # Prepare splits
        split_data = symbol_data[symbol_data['split_ratio'] != 1.0]
        if not split_data.empty:
            for date, row in split_data.iterrows():
                splits.append({
                    'sid': sid,
                    'ratio': row['split_ratio'],
                    'effective_date': date,
                })
        
        # Prepare dividends
        div_data = symbol_data[symbol_data['ex_dividend'] > 0]
        if not div_data.empty:
            for date, row in div_data.iterrows():
                dividends.append({
                    'sid': sid,
                    'amount': row['ex_dividend'],
                    'ex_date': date,
                    'record_date': date,
                    'declared_date': date,
                    'pay_date': date,
                })
    
    # Write daily bars
    if show_progress:
        log.info("Writing daily bars")
    
    def gen_daily_bars():
        for symbol in symbols:
            if symbol not in raw_data.index.get_level_values('symbol'):
                continue
            
            symbol_data = raw_data.loc[symbol]
            sid = asset_metadata[asset_metadata['symbol'] == symbol].index[0]
            
            # Ensure data is sorted by date
            symbol_data = symbol_data.sort_index()
            
            yield sid, symbol_data[['open', 'high', 'low', 'close', 'volume']]
    
    daily_bar_writer.write(gen_daily_bars(), show_progress=show_progress)
    
    # Write adjustments
    if splits or dividends:
        if show_progress:
            log.info("Writing adjustments")
        
        adjustment_writer.write(
            splits=pd.DataFrame(splits) if splits else None,
            dividends=pd.DataFrame(dividends) if dividends else None,
        )
    
    log.info("Yahoo bundle ingestion complete")

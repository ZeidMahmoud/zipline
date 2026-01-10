"""
Real-time data feed implementations for live trading.

This module provides data feed classes for real-time market data
from various sources including WebSocket-based feeds.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import logging
from threading import Thread, Event
import json

log = logging.getLogger(__name__)


class BarData:
    """Represents a bar of market data."""
    
    def __init__(self, symbol: str, timestamp: datetime, 
                 open: float, high: float, low: float, close: float, volume: int):
        self.symbol = symbol
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    
    def __repr__(self):
        return (f"BarData(symbol={self.symbol}, timestamp={self.timestamp}, "
                f"open={self.open}, high={self.high}, low={self.low}, "
                f"close={self.close}, volume={self.volume})")


class LiveDataFeed(ABC):
    """
    Abstract base class for live data feeds.
    
    All live data feed implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self):
        """Initialize the data feed."""
        self.connected = False
        self._callbacks: List[Callable[[BarData], None]] = []
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the data feed.
        
        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the data feed."""
        pass
    
    @abstractmethod
    def subscribe(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time data for symbols.
        
        Parameters
        ----------
        symbols : list of str
            List of symbols to subscribe to.
            
        Returns
        -------
        bool
            True if subscription successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def unsubscribe(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from real-time data for symbols.
        
        Parameters
        ----------
        symbols : list of str
            List of symbols to unsubscribe from.
            
        Returns
        -------
        bool
            True if unsubscription successful, False otherwise.
        """
        pass
    
    def register_callback(self, callback: Callable[[BarData], None]) -> None:
        """
        Register a callback for new bar data.
        
        Parameters
        ----------
        callback : callable
            Function to call when new bar data arrives.
            Should accept a BarData object.
        """
        self._callbacks.append(callback)
        log.info("Registered data callback")
    
    def _notify_callbacks(self, bar: BarData) -> None:
        """
        Notify all registered callbacks of new bar data.
        
        Parameters
        ----------
        bar : BarData
            The new bar data.
        """
        for callback in self._callbacks:
            try:
                callback(bar)
            except Exception as e:
                log.error(f"Error in data callback: {e}")


class WebSocketDataFeed(LiveDataFeed):
    """
    WebSocket-based real-time data feed.
    
    This implementation provides a generic WebSocket-based data feed
    that can be adapted for various data providers.
    
    Parameters
    ----------
    url : str
        WebSocket URL to connect to.
    auth_token : str, optional
        Authentication token if required.
    """
    
    def __init__(self, url: str, auth_token: Optional[str] = None):
        super().__init__()
        self.url = url
        self.auth_token = auth_token
        self._ws = None
        self._thread = None
        self._stop_event = Event()
        self._subscribed_symbols: List[str] = []
    
    def connect(self) -> bool:
        """
        Connect to the WebSocket data feed.
        
        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        try:
            import websockets
            self.connected = True
            log.info(f"WebSocket data feed initialized for {self.url}")
            return True
        except ImportError:
            log.error("websockets package not installed. Install with: pip install websockets")
            return False
        except Exception as e:
            log.error(f"Failed to initialize WebSocket connection: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the WebSocket data feed."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.connected = False
        log.info("Disconnected from WebSocket data feed")
    
    def subscribe(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time data for symbols.
        
        Parameters
        ----------
        symbols : list of str
            List of symbols to subscribe to.
            
        Returns
        -------
        bool
            True if subscription successful, False otherwise.
        """
        if not self.connected:
            log.error("Not connected to data feed")
            return False
        
        for symbol in symbols:
            if symbol not in self._subscribed_symbols:
                self._subscribed_symbols.append(symbol)
        
        log.info(f"Subscribed to symbols: {symbols}")
        return True
    
    def unsubscribe(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from real-time data for symbols.
        
        Parameters
        ----------
        symbols : list of str
            List of symbols to unsubscribe from.
            
        Returns
        -------
        bool
            True if unsubscription successful, False otherwise.
        """
        if not self.connected:
            log.error("Not connected to data feed")
            return False
        
        for symbol in symbols:
            if symbol in self._subscribed_symbols:
                self._subscribed_symbols.remove(symbol)
        
        log.info(f"Unsubscribed from symbols: {symbols}")
        return True
    
    def _parse_message(self, message: str) -> Optional[BarData]:
        """
        Parse a WebSocket message into BarData.
        
        This is a generic implementation that expects JSON messages
        with standard OHLCV fields. Override for specific providers.
        
        Parameters
        ----------
        message : str
            The WebSocket message to parse.
            
        Returns
        -------
        BarData or None
            Parsed bar data if successful, None otherwise.
        """
        try:
            data = json.loads(message)
            
            # Generic parsing - adjust based on actual data provider format
            return BarData(
                symbol=data.get('symbol', ''),
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                open=float(data.get('open', 0)),
                high=float(data.get('high', 0)),
                low=float(data.get('low', 0)),
                close=float(data.get('close', 0)),
                volume=int(data.get('volume', 0))
            )
        except Exception as e:
            log.error(f"Failed to parse message: {e}")
            return None
    
    def start_streaming(self) -> None:
        """Start streaming data in a background thread."""
        if self._thread and self._thread.is_alive():
            log.warning("Streaming already started")
            return
        
        self._stop_event.clear()
        self._thread = Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        log.info("Started streaming data")
    
    def _stream_loop(self) -> None:
        """
        Main streaming loop (runs in background thread).
        
        This is a placeholder implementation. Actual implementation
        would maintain WebSocket connection and process messages.
        """
        log.info("Streaming loop started (placeholder)")
        while not self._stop_event.is_set():
            self._stop_event.wait(1)
        log.info("Streaming loop stopped")


class AlpacaDataFeed(WebSocketDataFeed):
    """
    Alpaca-specific WebSocket data feed.
    
    Provides real-time market data from Alpaca.
    
    Parameters
    ----------
    api_key : str
        Alpaca API key.
    api_secret : str
        Alpaca API secret.
    """
    
    def __init__(self, api_key: str, api_secret: str):
        super().__init__(
            url='wss://stream.data.alpaca.markets/v2/iex',
            auth_token=None
        )
        self.api_key = api_key
        self.api_secret = api_secret
    
    def connect(self) -> bool:
        """
        Connect to Alpaca WebSocket feed.
        
        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        try:
            # Placeholder - actual implementation would use Alpaca's WebSocket API
            log.info("Connected to Alpaca data feed (placeholder)")
            self.connected = True
            return True
        except Exception as e:
            log.error(f"Failed to connect to Alpaca data feed: {e}")
            return False

"""
Broker integrations for live trading.

This module provides base classes and implementations for various broker
integrations, enabling live order routing and execution.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

log = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position in a security."""
    asset: Any
    amount: float
    cost_basis: float
    last_sale_price: float
    last_sale_date: datetime


@dataclass
class Order:
    """Represents an order."""
    id: str
    asset: Any
    amount: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled: float = 0.0
    commission: float = 0.0
    status: str = 'open'


class BaseBroker(ABC):
    """
    Abstract base class for all broker implementations.
    
    All broker integrations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self):
        """Initialize the broker connection."""
        self.connected = False
        
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the broker.
        
        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the broker."""
        pass
    
    @abstractmethod
    def submit_order(self, asset: Any, amount: float, 
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> Order:
        """
        Submit an order to the broker.
        
        Parameters
        ----------
        asset : Any
            The asset to trade.
        amount : float
            The number of shares/contracts to trade. Positive for buy, negative for sell.
        limit_price : float, optional
            Limit price for limit orders.
        stop_price : float, optional
            Stop price for stop orders.
            
        Returns
        -------
        Order
            The submitted order object.
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Parameters
        ----------
        order_id : str
            The ID of the order to cancel.
            
        Returns
        -------
        bool
            True if cancellation successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[Any, Position]:
        """
        Get current positions.
        
        Returns
        -------
        dict
            Dictionary mapping assets to Position objects.
        """
        pass
    
    @abstractmethod
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balance information.
        
        Returns
        -------
        dict
            Dictionary with keys: 'cash', 'portfolio_value', 'buying_power', etc.
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get the status of an order.
        
        Parameters
        ----------
        order_id : str
            The ID of the order.
            
        Returns
        -------
        Order or None
            The order object if found, None otherwise.
        """
        pass


class AlpacaBroker(BaseBroker):
    """
    Alpaca broker integration for commission-free trading.
    
    Requires alpaca-trade-api package.
    
    Parameters
    ----------
    api_key : str
        Alpaca API key.
    api_secret : str
        Alpaca API secret.
    base_url : str, optional
        Alpaca API base URL. Defaults to paper trading URL.
    """
    
    def __init__(self, api_key: str, api_secret: str, 
                 base_url: str = 'https://paper-api.alpaca.markets'):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self._api = None
        
    def connect(self) -> bool:
        """
        Establish connection to Alpaca.
        
        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        try:
            import alpaca_trade_api as tradeapi
            self._api = tradeapi.REST(
                self.api_key,
                self.api_secret,
                self.base_url,
                api_version='v2'
            )
            # Test connection
            self._api.get_account()
            self.connected = True
            log.info("Connected to Alpaca successfully")
            return True
        except ImportError:
            log.error("alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")
            return False
        except Exception as e:
            log.error(f"Failed to connect to Alpaca: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        self._api = None
        self.connected = False
        log.info("Disconnected from Alpaca")
    
    def submit_order(self, asset: Any, amount: float,
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> Order:
        """
        Submit an order to Alpaca.
        
        Parameters
        ----------
        asset : Any
            The asset to trade.
        amount : float
            The number of shares to trade. Positive for buy, negative for sell.
        limit_price : float, optional
            Limit price for limit orders.
        stop_price : float, optional
            Stop price for stop orders.
            
        Returns
        -------
        Order
            The submitted order object.
        """
        if not self.connected:
            raise RuntimeError("Not connected to broker")
        
        symbol = str(asset.symbol) if hasattr(asset, 'symbol') else str(asset)
        side = 'buy' if amount > 0 else 'sell'
        qty = abs(amount)
        
        # Determine order type
        if limit_price and stop_price:
            order_type = 'stop_limit'
        elif limit_price:
            order_type = 'limit'
        elif stop_price:
            order_type = 'stop'
        else:
            order_type = 'market'
        
        try:
            alpaca_order = self._api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force='day',
                limit_price=limit_price,
                stop_price=stop_price
            )
            
            order = Order(
                id=alpaca_order.id,
                asset=asset,
                amount=amount,
                limit_price=limit_price,
                stop_price=stop_price,
                filled=0.0,
                commission=0.0,  # Alpaca is commission-free
                status=alpaca_order.status
            )
            
            log.info(f"Submitted order: {order}")
            return order
            
        except Exception as e:
            log.error(f"Failed to submit order: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Parameters
        ----------
        order_id : str
            The ID of the order to cancel.
            
        Returns
        -------
        bool
            True if cancellation successful, False otherwise.
        """
        if not self.connected:
            raise RuntimeError("Not connected to broker")
        
        try:
            self._api.cancel_order(order_id)
            log.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            log.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_positions(self) -> Dict[Any, Position]:
        """
        Get current positions from Alpaca.
        
        Returns
        -------
        dict
            Dictionary mapping assets to Position objects.
        """
        if not self.connected:
            raise RuntimeError("Not connected to broker")
        
        try:
            alpaca_positions = self._api.list_positions()
            positions = {}
            
            for pos in alpaca_positions:
                # Would need to map symbol back to asset object
                positions[pos.symbol] = Position(
                    asset=pos.symbol,
                    amount=float(pos.qty),
                    cost_basis=float(pos.cost_basis),
                    last_sale_price=float(pos.current_price),
                    last_sale_date=datetime.now()
                )
            
            return positions
            
        except Exception as e:
            log.error(f"Failed to get positions: {e}")
            return {}
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balance information from Alpaca.
        
        Returns
        -------
        dict
            Dictionary with keys: 'cash', 'portfolio_value', 'buying_power', etc.
        """
        if not self.connected:
            raise RuntimeError("Not connected to broker")
        
        try:
            account = self._api.get_account()
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
            }
        except Exception as e:
            log.error(f"Failed to get account balance: {e}")
            return {}
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get the status of an order.
        
        Parameters
        ----------
        order_id : str
            The ID of the order.
            
        Returns
        -------
        Order or None
            The order object if found, None otherwise.
        """
        if not self.connected:
            raise RuntimeError("Not connected to broker")
        
        try:
            alpaca_order = self._api.get_order(order_id)
            
            order = Order(
                id=alpaca_order.id,
                asset=alpaca_order.symbol,  # Would need proper asset mapping
                amount=float(alpaca_order.qty) if alpaca_order.side == 'buy' else -float(alpaca_order.qty),
                limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                filled=float(alpaca_order.filled_qty),
                commission=0.0,
                status=alpaca_order.status
            )
            
            return order
            
        except Exception as e:
            log.error(f"Failed to get order status for {order_id}: {e}")
            return None


class IBBroker(BaseBroker):
    """
    Interactive Brokers integration (stub/interface).
    
    This is a placeholder for Interactive Brokers integration.
    Actual implementation would require ib_insync or similar package.
    
    Parameters
    ----------
    host : str
        IB Gateway/TWS host address.
    port : int
        IB Gateway/TWS port.
    client_id : int
        Unique client ID.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        super().__init__()
        self.host = host
        self.port = port
        self.client_id = client_id
        
    def connect(self) -> bool:
        """
        Establish connection to Interactive Brokers.
        
        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        log.warning("IBBroker is a stub implementation. Full IB integration not implemented.")
        return False
    
    def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        pass
    
    def submit_order(self, asset: Any, amount: float,
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> Order:
        """Submit an order to IB (stub)."""
        raise NotImplementedError("IBBroker is a stub implementation")
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order (stub)."""
        raise NotImplementedError("IBBroker is a stub implementation")
    
    def get_positions(self) -> Dict[Any, Position]:
        """Get positions (stub)."""
        raise NotImplementedError("IBBroker is a stub implementation")
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance (stub)."""
        raise NotImplementedError("IBBroker is a stub implementation")
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status (stub)."""
        raise NotImplementedError("IBBroker is a stub implementation")

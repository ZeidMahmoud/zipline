"""
Live execution engine for real-time order routing and execution.

This module provides the execution engine that handles live trading,
including order routing, status tracking, and execution callbacks.
"""
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime
import logging
from threading import Lock
from enum import Enum

from .broker import BaseBroker, Order

log = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Enumeration of order statuses."""
    OPEN = 'open'
    FILLED = 'filled'
    PARTIALLY_FILLED = 'partially_filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'


class LiveExecutionEngine:
    """
    Live execution engine for real-time order routing and execution.
    
    This engine manages the lifecycle of orders from submission through
    execution, providing status tracking and callbacks for order events.
    
    Parameters
    ----------
    broker : BaseBroker
        The broker instance to use for order execution.
    
    Attributes
    ----------
    broker : BaseBroker
        The connected broker instance.
    orders : dict
        Dictionary mapping order IDs to Order objects.
    """
    
    def __init__(self, broker: BaseBroker):
        """
        Initialize the live execution engine.
        
        Parameters
        ----------
        broker : BaseBroker
            The broker instance to use for order execution.
        """
        self.broker = broker
        self.orders: Dict[str, Order] = {}
        self._lock = Lock()
        self._callbacks: Dict[str, List[Callable]] = {
            'on_order_filled': [],
            'on_order_cancelled': [],
            'on_order_rejected': [],
            'on_order_partially_filled': [],
        }
        
    def connect(self) -> bool:
        """
        Connect to the broker.
        
        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        return self.broker.connect()
    
    def disconnect(self) -> None:
        """Disconnect from the broker."""
        self.broker.disconnect()
    
    def submit_market_order(self, asset: Any, amount: float) -> Order:
        """
        Submit a market order.
        
        Parameters
        ----------
        asset : Any
            The asset to trade.
        amount : float
            The number of shares/contracts to trade.
            Positive for buy, negative for sell.
            
        Returns
        -------
        Order
            The submitted order object.
        """
        with self._lock:
            order = self.broker.submit_order(asset, amount)
            self.orders[order.id] = order
            log.info(f"Submitted market order: {order.id} for {asset}")
            return order
    
    def submit_limit_order(self, asset: Any, amount: float, 
                          limit_price: float) -> Order:
        """
        Submit a limit order.
        
        Parameters
        ----------
        asset : Any
            The asset to trade.
        amount : float
            The number of shares/contracts to trade.
            Positive for buy, negative for sell.
        limit_price : float
            The limit price for the order.
            
        Returns
        -------
        Order
            The submitted order object.
        """
        with self._lock:
            order = self.broker.submit_order(asset, amount, limit_price=limit_price)
            self.orders[order.id] = order
            log.info(f"Submitted limit order: {order.id} for {asset} at {limit_price}")
            return order
    
    def submit_stop_order(self, asset: Any, amount: float, 
                         stop_price: float) -> Order:
        """
        Submit a stop order.
        
        Parameters
        ----------
        asset : Any
            The asset to trade.
        amount : float
            The number of shares/contracts to trade.
            Positive for buy, negative for sell.
        stop_price : float
            The stop price for the order.
            
        Returns
        -------
        Order
            The submitted order object.
        """
        with self._lock:
            order = self.broker.submit_order(asset, amount, stop_price=stop_price)
            self.orders[order.id] = order
            log.info(f"Submitted stop order: {order.id} for {asset} at {stop_price}")
            return order
    
    def submit_stop_limit_order(self, asset: Any, amount: float,
                               stop_price: float, limit_price: float) -> Order:
        """
        Submit a stop-limit order.
        
        Parameters
        ----------
        asset : Any
            The asset to trade.
        amount : float
            The number of shares/contracts to trade.
            Positive for buy, negative for sell.
        stop_price : float
            The stop price for the order.
        limit_price : float
            The limit price for the order.
            
        Returns
        -------
        Order
            The submitted order object.
        """
        with self._lock:
            order = self.broker.submit_order(
                asset, amount, 
                limit_price=limit_price,
                stop_price=stop_price
            )
            self.orders[order.id] = order
            log.info(f"Submitted stop-limit order: {order.id} for {asset}")
            return order
    
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
        with self._lock:
            if order_id not in self.orders:
                log.warning(f"Order {order_id} not found")
                return False
            
            success = self.broker.cancel_order(order_id)
            if success:
                order = self.orders[order_id]
                order.status = OrderStatus.CANCELLED.value
                self._trigger_callbacks('on_order_cancelled', order)
                log.info(f"Cancelled order: {order_id}")
            return success
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get an order by ID.
        
        Parameters
        ----------
        order_id : str
            The ID of the order.
            
        Returns
        -------
        Order or None
            The order object if found, None otherwise.
        """
        return self.orders.get(order_id)
    
    def get_open_orders(self) -> List[Order]:
        """
        Get all open orders.
        
        Returns
        -------
        list of Order
            List of open orders.
        """
        return [
            order for order in self.orders.values()
            if order.status == OrderStatus.OPEN.value
        ]
    
    def update_order_status(self, order_id: str) -> Optional[Order]:
        """
        Update the status of an order from the broker.
        
        Parameters
        ----------
        order_id : str
            The ID of the order to update.
            
        Returns
        -------
        Order or None
            The updated order object if found, None otherwise.
        """
        with self._lock:
            broker_order = self.broker.get_order_status(order_id)
            if broker_order is None:
                log.warning(f"Could not fetch status for order {order_id}")
                return None
            
            if order_id in self.orders:
                old_status = self.orders[order_id].status
                self.orders[order_id] = broker_order
                
                # Trigger callbacks based on status change
                if old_status != broker_order.status:
                    if broker_order.status == OrderStatus.FILLED.value:
                        self._trigger_callbacks('on_order_filled', broker_order)
                    elif broker_order.status == OrderStatus.PARTIALLY_FILLED.value:
                        self._trigger_callbacks('on_order_partially_filled', broker_order)
                    elif broker_order.status == OrderStatus.CANCELLED.value:
                        self._trigger_callbacks('on_order_cancelled', broker_order)
                    elif broker_order.status == OrderStatus.REJECTED.value:
                        self._trigger_callbacks('on_order_rejected', broker_order)
            
            return broker_order
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for order events.
        
        Parameters
        ----------
        event : str
            The event name. One of: 'on_order_filled', 'on_order_cancelled',
            'on_order_rejected', 'on_order_partially_filled'.
        callback : callable
            The callback function to register. Should accept an Order object.
        """
        if event not in self._callbacks:
            raise ValueError(f"Unknown event: {event}")
        self._callbacks[event].append(callback)
        log.info(f"Registered callback for event: {event}")
    
    def _trigger_callbacks(self, event: str, order: Order) -> None:
        """
        Trigger all callbacks for an event.
        
        Parameters
        ----------
        event : str
            The event name.
        order : Order
            The order object to pass to callbacks.
        """
        for callback in self._callbacks.get(event, []):
            try:
                callback(order)
            except Exception as e:
                log.error(f"Error in callback for {event}: {e}")
    
    def get_positions(self) -> Dict[Any, Any]:
        """
        Get current positions from the broker.
        
        Returns
        -------
        dict
            Dictionary mapping assets to Position objects.
        """
        return self.broker.get_positions()
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balance information from the broker.
        
        Returns
        -------
        dict
            Dictionary with account balance information.
        """
        return self.broker.get_account_balance()

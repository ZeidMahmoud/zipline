"""
FastAPI application for Zipline dashboard.

This module provides the main FastAPI application for the monitoring dashboard.
"""
from typing import Optional, Dict, Any
import logging

log = logging.getLogger(__name__)

try:
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from pathlib import Path
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None
    FASTAPI_AVAILABLE = False
    log.warning("FastAPI not installed. Install with: pip install 'zipline[dashboard]'")


class DashboardApp:
    """
    Zipline Dashboard Application.
    
    Provides a web-based interface for monitoring live trading,
    viewing performance metrics, positions, and risk analytics.
    
    Parameters
    ----------
    title : str, optional
        Dashboard title. Default is 'Zipline Dashboard'.
    host : str, optional
        Host address to bind to. Default is '0.0.0.0'.
    port : int, optional
        Port to bind to. Default is 8000.
    
    Examples
    --------
    >>> from zipline.dashboard import DashboardApp
    >>> app = DashboardApp()
    >>> # Run with: uvicorn app:app
    """
    
    def __init__(self, title: str = 'Zipline Dashboard',
                 host: str = '0.0.0.0', port: int = 8000):
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for dashboard. "
                "Install with: pip install 'zipline[dashboard]'"
            )
        
        self.title = title
        self.host = host
        self.port = port
        self.app = None
        self._performance_data = {}
        self._positions = {}
        self._orders = []
        self._risk_metrics = {}
    
    def create_app(self) -> FastAPI:
        """
        Create the FastAPI application.
        
        Returns
        -------
        FastAPI
            The configured FastAPI app.
        """
        from .routes import router
        
        app = FastAPI(title=self.title)
        
        # Get the directory of this file
        dashboard_dir = Path(__file__).parent
        
        # Mount static files
        static_dir = dashboard_dir / 'static'
        if static_dir.exists():
            app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')
        
        # Setup templates
        templates_dir = dashboard_dir / 'templates'
        if templates_dir.exists():
            app.state.templates = Jinja2Templates(directory=str(templates_dir))
        
        # Include routers
        app.include_router(router, prefix='/api')
        
        # Store reference to dashboard for accessing data
        app.state.dashboard = self
        
        self.app = app
        return app
    
    def update_performance(self, data: Dict[str, Any]) -> None:
        """
        Update performance data.
        
        Parameters
        ----------
        data : dict
            Performance data including returns, Sharpe ratio, etc.
        """
        self._performance_data.update(data)
    
    def update_positions(self, positions: Dict[str, Any]) -> None:
        """
        Update current positions.
        
        Parameters
        ----------
        positions : dict
            Current positions data.
        """
        self._positions = positions
    
    def add_order(self, order: Dict[str, Any]) -> None:
        """
        Add an order to history.
        
        Parameters
        ----------
        order : dict
            Order data.
        """
        self._orders.append(order)
    
    def update_risk_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update risk metrics.
        
        Parameters
        ----------
        metrics : dict
            Risk metrics including VaR, drawdown, etc.
        """
        self._risk_metrics.update(metrics)
    
    def get_performance(self) -> Dict[str, Any]:
        """Get current performance data."""
        return self._performance_data
    
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        return self._positions
    
    def get_orders(self, limit: int = 100) -> list:
        """
        Get order history.
        
        Parameters
        ----------
        limit : int, optional
            Maximum number of orders to return.
            
        Returns
        -------
        list
            Recent orders.
        """
        return self._orders[-limit:]
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        return self._risk_metrics
    
    def run(self):
        """
        Run the dashboard server.
        
        This is a convenience method for development.
        For production, use uvicorn directly.
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "uvicorn is required to run dashboard. "
                "Install with: pip install uvicorn"
            )
        
        if self.app is None:
            self.create_app()
        
        log.info(f"Starting dashboard on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)


def create_app(title: str = 'Zipline Dashboard') -> FastAPI:
    """
    Create a Zipline dashboard application.
    
    This is a factory function for creating the FastAPI app.
    
    Parameters
    ----------
    title : str, optional
        Dashboard title.
        
    Returns
    -------
    FastAPI
        The configured application.
        
    Examples
    --------
    >>> app = create_app()
    >>> # Run with: uvicorn module:app
    """
    dashboard = DashboardApp(title=title)
    return dashboard.create_app()

"""
API routes for Zipline dashboard.

This module defines the REST API endpoints for the dashboard.
"""
from typing import Dict, Any, Optional
import logging

log = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, Request, WebSocket
    from fastapi.responses import HTMLResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    APIRouter = object
    FASTAPI_AVAILABLE = False

# Create router
if FASTAPI_AVAILABLE:
    router = APIRouter()
else:
    router = None


if FASTAPI_AVAILABLE:
    
    @router.get("/")
    async def root():
        """Root endpoint."""
        return {"message": "Zipline Dashboard API", "version": "1.0"}
    
    @router.get("/performance", response_model=Dict[str, Any])
    async def get_performance(request: Request):
        """
        Get performance metrics.
        
        Returns current algorithm performance including returns,
        Sharpe ratio, and other metrics.
        """
        dashboard = request.app.state.dashboard
        performance = dashboard.get_performance()
        
        if not performance:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }
        
        return performance
    
    @router.get("/positions", response_model=Dict[str, Any])
    async def get_positions(request: Request):
        """
        Get current positions.
        
        Returns all current open positions with cost basis,
        current value, and P&L.
        """
        dashboard = request.app.state.dashboard
        positions = dashboard.get_positions()
        
        return positions
    
    @router.get("/orders")
    async def get_orders(request: Request, limit: int = 100):
        """
        Get order history.
        
        Parameters
        ----------
        limit : int, optional
            Maximum number of orders to return. Default is 100.
            
        Returns recent orders with status and fill information.
        """
        dashboard = request.app.state.dashboard
        orders = dashboard.get_orders(limit=limit)
        
        return {"orders": orders, "count": len(orders)}
    
    @router.get("/risk", response_model=Dict[str, Any])
    async def get_risk_metrics(request: Request):
        """
        Get risk metrics.
        
        Returns risk metrics including VaR, volatility,
        drawdown, and exposure metrics.
        """
        dashboard = request.app.state.dashboard
        risk_metrics = dashboard.get_risk_metrics()
        
        if not risk_metrics:
            return {
                "var_95": 0.0,
                "var_99": 0.0,
                "volatility": 0.0,
                "max_drawdown": 0.0,
                "current_drawdown": 0.0,
                "beta": 0.0,
            }
        
        return risk_metrics
    
    @router.get("/health")
    async def health_check():
        """
        Health check endpoint.
        
        Returns server status and uptime information.
        """
        return {
            "status": "healthy",
            "service": "zipline-dashboard",
        }
    
    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """
        WebSocket endpoint for real-time updates.
        
        Clients can connect to receive real-time performance
        and position updates.
        """
        await websocket.accept()
        
        try:
            while True:
                # Wait for client messages
                data = await websocket.receive_text()
                
                # Echo back for now (placeholder)
                await websocket.send_text(f"Echo: {data}")
                
        except Exception as e:
            log.error(f"WebSocket error: {e}")
        finally:
            await websocket.close()
    
    @router.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_page(request: Request):
        """
        Main dashboard page.
        
        Renders the HTML dashboard interface.
        """
        if hasattr(request.app.state, 'templates'):
            return request.app.state.templates.TemplateResponse(
                "dashboard.html",
                {"request": request, "title": "Zipline Dashboard"}
            )
        else:
            return HTMLResponse(content="""
                <html>
                    <head><title>Zipline Dashboard</title></head>
                    <body>
                        <h1>Zipline Dashboard</h1>
                        <p>API endpoints:</p>
                        <ul>
                            <li><a href="/api/performance">/api/performance</a></li>
                            <li><a href="/api/positions">/api/positions</a></li>
                            <li><a href="/api/orders">/api/orders</a></li>
                            <li><a href="/api/risk">/api/risk</a></li>
                            <li><a href="/api/health">/api/health</a></li>
                        </ul>
                    </body>
                </html>
            """)

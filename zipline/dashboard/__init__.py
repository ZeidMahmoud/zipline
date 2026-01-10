"""
Web dashboard for Zipline live trading and monitoring.

This module provides a FastAPI-based web dashboard for monitoring
trading performance, positions, and risk metrics in real-time.
"""
from .app import create_app, DashboardApp
from .routes import router

__all__ = [
    'create_app',
    'DashboardApp',
    'router',
]

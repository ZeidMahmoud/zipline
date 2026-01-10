"""Stress Testing Framework."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class StressTestFramework:
    """Scenario-based stress testing."""
    
    def __init__(self):
        self.scenarios = {}
    
    def add_scenario(self, name, shocks):
        """Add a stress test scenario."""
        self.scenarios[name] = shocks
    
    def run_scenario(self, portfolio_weights, asset_returns, scenario_name):
        """Run a stress test scenario."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario {scenario_name} not found")
        
        shocks = self.scenarios[scenario_name]
        stressed_returns = asset_returns + shocks
        portfolio_return = portfolio_weights @ stressed_returns
        
        return portfolio_return

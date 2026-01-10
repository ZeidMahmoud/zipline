"""Drawdown Analysis."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class DrawdownAnalyzer:
    """Maximum drawdown and recovery analysis."""
    
    def __init__(self):
        pass
    
    def calculate_drawdown(self, returns):
        """Calculate drawdown series."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    def max_drawdown(self, returns):
        """Calculate maximum drawdown."""
        drawdown = self.calculate_drawdown(returns)
        return np.min(drawdown)
    
    def drawdown_duration(self, returns):
        """Calculate drawdown duration."""
        drawdown = self.calculate_drawdown(returns)
        in_drawdown = drawdown < 0
        
        durations = []
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        return max(durations) if durations else 0

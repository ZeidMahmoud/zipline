"""
Ensemble Methods for Machine Learning in Trading.

Provides stacking, voting, and boosting ensemble methods.
"""

try:
    from .stacking import StackingPredictor
    from .voting import VotingPredictor
    from .boosting import GradientBoostingPredictor
    
    __all__ = [
        'StackingPredictor',
        'VotingPredictor',
        'GradientBoostingPredictor',
    ]
except ImportError:
    __all__ = []

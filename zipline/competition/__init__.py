"""
Zipline Backtesting Competition Platform

This module provides a comprehensive competition platform for algorithmic trading strategies,
including leaderboards, submissions, evaluations, and prize management.
"""

from .platform import CompetitionPlatform
from .leaderboard import Leaderboard
from .submission import StrategySubmission
from .evaluation import CompetitionEvaluator
from .prizes import PrizePool

__all__ = [
    'CompetitionPlatform',
    'Leaderboard',
    'StrategySubmission',
    'CompetitionEvaluator',
    'PrizePool',
]

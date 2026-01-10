"""
Leaderboard System - Track and display competition rankings
"""
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class RankingMetric(Enum):
    """Metrics used for ranking participants"""
    RETURNS = "returns"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    ALPHA = "alpha"
    BETA = "beta"
    WIN_RATE = "win_rate"


class Leaderboard:
    """
    Track and manage competition leaderboards.
    
    Supports multiple ranking metrics, historical snapshots,
    percentile rankings, and achievement badges.
    
    Parameters
    ----------
    competition_id : str
        ID of the competition
    primary_metric : RankingMetric, optional
        Primary metric for ranking (default: SHARPE_RATIO)
    
    Examples
    --------
    >>> leaderboard = Leaderboard("comp_123")
    >>> leaderboard.update_score("user_1", {"sharpe_ratio": 2.5, "returns": 0.15})
    >>> rankings = leaderboard.get_rankings()
    """
    
    def __init__(
        self,
        competition_id: str,
        primary_metric: RankingMetric = RankingMetric.SHARPE_RATIO,
    ):
        self.competition_id = competition_id
        self.primary_metric = primary_metric
        self.scores: Dict[str, Dict[str, float]] = {}
        self.snapshots: List[Dict[str, Any]] = []
        self.badges: Dict[str, List[str]] = {}
    
    def update_score(
        self,
        user_id: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        Update a participant's scores.
        
        Parameters
        ----------
        user_id : str
            User ID
        metrics : dict
            Dictionary of metric name to value
        """
        if user_id not in self.scores:
            self.scores[user_id] = {}
        
        self.scores[user_id].update(metrics)
        self.scores[user_id]['updated_at'] = datetime.now().isoformat()
    
    def get_rankings(
        self,
        metric: Optional[RankingMetric] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get current rankings.
        
        Parameters
        ----------
        metric : RankingMetric, optional
            Metric to rank by (default: primary_metric)
        limit : int, optional
            Maximum number of results to return
            
        Returns
        -------
        list of dict
            Ranked list of participants with scores
        """
        if metric is None:
            metric = self.primary_metric
        
        metric_name = metric.value
        
        # Filter participants who have the metric
        ranked = [
            {
                'user_id': user_id,
                'rank': 0,
                'score': scores.get(metric_name, 0.0),
                'metrics': scores,
            }
            for user_id, scores in self.scores.items()
            if metric_name in scores
        ]
        
        # Sort by metric (higher is better for most metrics except drawdown)
        reverse = metric != RankingMetric.MAX_DRAWDOWN
        ranked.sort(key=lambda x: x['score'], reverse=reverse)
        
        # Assign ranks
        for i, entry in enumerate(ranked, 1):
            entry['rank'] = i
        
        if limit:
            ranked = ranked[:limit]
        
        return ranked
    
    def get_user_rank(
        self,
        user_id: str,
        metric: Optional[RankingMetric] = None,
    ) -> Optional[int]:
        """
        Get a user's current rank.
        
        Parameters
        ----------
        user_id : str
            User ID
        metric : RankingMetric, optional
            Metric to rank by
            
        Returns
        -------
        int or None
            User's rank (1-indexed) or None if not found
        """
        rankings = self.get_rankings(metric)
        for entry in rankings:
            if entry['user_id'] == user_id:
                return entry['rank']
        return None
    
    def get_percentile(
        self,
        user_id: str,
        metric: Optional[RankingMetric] = None,
    ) -> Optional[float]:
        """
        Get a user's percentile rank.
        
        Parameters
        ----------
        user_id : str
            User ID
        metric : RankingMetric, optional
            Metric to rank by
            
        Returns
        -------
        float or None
            Percentile (0-100) or None if not found
        """
        rankings = self.get_rankings(metric)
        if not rankings:
            return None
        
        for i, entry in enumerate(rankings):
            if entry['user_id'] == user_id:
                return (1 - (i / len(rankings))) * 100
        
        return None
    
    def create_snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of current leaderboard state.
        
        Returns
        -------
        dict
            Snapshot containing timestamp and rankings
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'rankings': self.get_rankings(),
            'metric': self.primary_metric.value,
        }
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_snapshots(self) -> List[Dict[str, Any]]:
        """
        Get all historical snapshots.
        
        Returns
        -------
        list of dict
            Historical leaderboard snapshots
        """
        return self.snapshots
    
    def award_badge(self, user_id: str, badge: str) -> None:
        """
        Award a badge to a user.
        
        Parameters
        ----------
        user_id : str
            User ID
        badge : str
            Badge name/identifier
        """
        if user_id not in self.badges:
            self.badges[user_id] = []
        
        if badge not in self.badges[user_id]:
            self.badges[user_id].append(badge)
    
    def get_badges(self, user_id: str) -> List[str]:
        """
        Get all badges for a user.
        
        Parameters
        ----------
        user_id : str
            User ID
            
        Returns
        -------
        list of str
            List of badge names
        """
        return self.badges.get(user_id, [])
    
    def get_top_n(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N participants.
        
        Parameters
        ----------
        n : int
            Number of top participants to return
            
        Returns
        -------
        list of dict
            Top N ranked participants
        """
        return self.get_rankings(limit=n)

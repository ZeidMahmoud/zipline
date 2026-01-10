"""
Competition REST API - FastAPI endpoints for competition management
"""
from typing import Dict, List, Optional, Any
from datetime import datetime

# Note: FastAPI is an optional dependency
# This module provides a minimal interface that can be extended


class CompetitionAPI:
    """
    REST API interface for competition management.
    
    This is a minimal implementation that provides the API structure.
    For full functionality, install with: pip install zipline[competition]
    
    Parameters
    ----------
    platform : CompetitionPlatform
        Competition platform instance
    """
    
    def __init__(self, platform):
        self.platform = platform
    
    def list_competitions(
        self,
        status: Optional[str] = None,
        competition_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List competitions.
        
        GET /api/competitions
        
        Parameters
        ----------
        status : str, optional
            Filter by status
        competition_type : str, optional
            Filter by type
        limit : int
            Maximum results
        offset : int
            Result offset
            
        Returns
        -------
        dict
            Response with competitions list
        """
        competitions = self.platform.list_competitions()
        
        # Apply filters
        if status:
            competitions = [c for c in competitions if c['status'].value == status]
        if competition_type:
            competitions = [c for c in competitions if c['type'].value == competition_type]
        
        # Apply pagination
        total = len(competitions)
        competitions = competitions[offset:offset+limit]
        
        return {
            'competitions': competitions,
            'total': total,
            'limit': limit,
            'offset': offset,
        }
    
    def get_competition(self, competition_id: str) -> Dict[str, Any]:
        """
        Get competition details.
        
        GET /api/competitions/{id}
        
        Parameters
        ----------
        competition_id : str
            Competition ID
            
        Returns
        -------
        dict
            Competition details
        """
        comp = self.platform.get_competition(competition_id)
        if not comp:
            return {'error': 'Competition not found', 'status': 404}
        
        return {'competition': comp, 'status': 200}
    
    def submit_strategy(
        self,
        competition_id: str,
        user_id: str,
        strategy_code: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit strategy to competition.
        
        POST /api/competitions/{id}/submit
        
        Parameters
        ----------
        competition_id : str
            Competition ID
        user_id : str
            User ID
        strategy_code : str
            Strategy code
        metadata : dict, optional
            Additional metadata
            
        Returns
        -------
        dict
            Submission result
        """
        from .submission import StrategySubmission
        
        # Create submission
        submission = StrategySubmission(
            competition_id=competition_id,
            user_id=user_id,
            strategy_code=strategy_code,
            metadata=metadata,
        )
        
        # Validate
        is_valid = submission.validate()
        
        if not is_valid:
            return {
                'status': 400,
                'error': 'Validation failed',
                'errors': submission.validation_errors,
            }
        
        return {
            'status': 201,
            'submission': submission.to_dict(),
        }
    
    def get_leaderboard(
        self,
        competition_id: str,
        metric: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Get competition leaderboard.
        
        GET /api/competitions/{id}/leaderboard
        
        Parameters
        ----------
        competition_id : str
            Competition ID
        metric : str, optional
            Ranking metric
        limit : int
            Maximum results
            
        Returns
        -------
        dict
            Leaderboard data
        """
        from .leaderboard import Leaderboard, RankingMetric
        
        # In a real implementation, this would fetch from storage
        leaderboard = Leaderboard(competition_id)
        
        # Convert metric string to enum if provided
        rank_metric = None
        if metric:
            try:
                rank_metric = RankingMetric(metric)
            except ValueError:
                pass
        
        rankings = leaderboard.get_rankings(metric=rank_metric, limit=limit)
        
        return {
            'status': 200,
            'competition_id': competition_id,
            'rankings': rankings,
            'metric': metric or leaderboard.primary_metric.value,
        }
    
    def get_results(
        self,
        competition_id: str,
    ) -> Dict[str, Any]:
        """
        Get competition results.
        
        GET /api/competitions/{id}/results
        
        Parameters
        ----------
        competition_id : str
            Competition ID
            
        Returns
        -------
        dict
            Competition results including winners and prizes
        """
        comp = self.platform.get_competition(competition_id)
        if not comp:
            return {'error': 'Competition not found', 'status': 404}
        
        from .leaderboard import Leaderboard
        from .prizes import PrizePool
        
        # Get final leaderboard
        leaderboard = Leaderboard(competition_id)
        rankings = leaderboard.get_rankings()
        
        # Get prize distribution
        prize_pool = PrizePool(competition_id)
        awards = prize_pool.distribute_prizes(rankings)
        
        return {
            'status': 200,
            'competition_id': competition_id,
            'rankings': rankings,
            'awards': awards,
        }


def create_fastapi_app(platform):
    """
    Create FastAPI application for competition platform.
    
    This requires FastAPI to be installed: pip install zipline[competition]
    
    Parameters
    ----------
    platform : CompetitionPlatform
        Competition platform instance
        
    Returns
    -------
    FastAPI
        FastAPI application
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI is required for the competition API. "
            "Install with: pip install zipline[competition]"
        )
    
    app = FastAPI(title="Zipline Competition API")
    api = CompetitionAPI(platform)
    
    # Define request models
    class StrategySubmit(BaseModel):
        user_id: str
        strategy_code: str
        metadata: Optional[Dict[str, Any]] = None
    
    @app.get("/api/competitions")
    def list_competitions(
        status: Optional[str] = None,
        competition_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ):
        return api.list_competitions(status, competition_type, limit, offset)
    
    @app.get("/api/competitions/{competition_id}")
    def get_competition(competition_id: str):
        result = api.get_competition(competition_id)
        if 'error' in result:
            raise HTTPException(status_code=result['status'], detail=result['error'])
        return result
    
    @app.post("/api/competitions/{competition_id}/submit")
    def submit_strategy(competition_id: str, submission: StrategySubmit):
        result = api.submit_strategy(
            competition_id,
            submission.user_id,
            submission.strategy_code,
            submission.metadata,
        )
        if result['status'] != 201:
            raise HTTPException(status_code=result['status'], detail=result.get('error'))
        return result
    
    @app.get("/api/competitions/{competition_id}/leaderboard")
    def get_leaderboard(
        competition_id: str,
        metric: Optional[str] = None,
        limit: int = 100,
    ):
        return api.get_leaderboard(competition_id, metric, limit)
    
    @app.get("/api/competitions/{competition_id}/results")
    def get_results(competition_id: str):
        result = api.get_results(competition_id)
        if 'error' in result:
            raise HTTPException(status_code=result['status'], detail=result['error'])
        return result
    
    return app

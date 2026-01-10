"""
Competition Platform - Main competition management class
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum


class CompetitionType(Enum):
    """Types of competitions"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class CompetitionStatus(Enum):
    """Competition status"""
    UPCOMING = "upcoming"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CompetitionPlatform:
    """
    Main platform class for managing backtesting competitions.
    
    This class handles creating competitions, managing participants,
    tracking submissions, and maintaining real-time rankings.
    
    Parameters
    ----------
    name : str
        Name of the competition platform
    database_url : str, optional
        Database connection URL for storing competition data
    cache_backend : str, optional
        Cache backend for real-time updates (default: 'memory')
        
    Examples
    --------
    >>> platform = CompetitionPlatform("Zipline Competitions")
    >>> comp = platform.create_competition(
    ...     name="Monthly Alpha Challenge",
    ...     competition_type=CompetitionType.MONTHLY,
    ...     start_date=datetime.now(),
    ...     end_date=datetime.now() + timedelta(days=30)
    ... )
    """
    
    def __init__(
        self,
        name: str,
        database_url: Optional[str] = None,
        cache_backend: str = 'memory'
    ):
        self.name = name
        self.database_url = database_url
        self.cache_backend = cache_backend
        self.competitions: Dict[str, Dict[str, Any]] = {}
        self._next_id = 1
    
    def create_competition(
        self,
        name: str,
        competition_type: CompetitionType,
        start_date: datetime,
        end_date: datetime,
        description: str = "",
        rules: Optional[Dict[str, Any]] = None,
        prize_pool: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new competition.
        
        Parameters
        ----------
        name : str
            Competition name
        competition_type : CompetitionType
            Type of competition (daily, weekly, monthly, custom)
        start_date : datetime
            Competition start date
        end_date : datetime
            Competition end date
        description : str, optional
            Competition description
        rules : dict, optional
            Competition rules and constraints
        prize_pool : dict, optional
            Prize pool configuration
            
        Returns
        -------
        str
            Competition ID
        """
        comp_id = f"comp_{self._next_id}"
        self._next_id += 1
        
        if rules is None:
            rules = {}
        
        self.competitions[comp_id] = {
            'id': comp_id,
            'name': name,
            'type': competition_type,
            'start_date': start_date,
            'end_date': end_date,
            'description': description,
            'rules': rules,
            'prize_pool': prize_pool,
            'status': CompetitionStatus.UPCOMING,
            'participants': [],
            'submissions': {},
            'created_at': datetime.now(),
        }
        
        return comp_id
    
    def get_competition(self, comp_id: str) -> Optional[Dict[str, Any]]:
        """
        Get competition details.
        
        Parameters
        ----------
        comp_id : str
            Competition ID
            
        Returns
        -------
        dict or None
            Competition details if found
        """
        return self.competitions.get(comp_id)
    
    def list_competitions(
        self,
        status: Optional[CompetitionStatus] = None,
        competition_type: Optional[CompetitionType] = None,
    ) -> List[Dict[str, Any]]:
        """
        List competitions with optional filtering.
        
        Parameters
        ----------
        status : CompetitionStatus, optional
            Filter by competition status
        competition_type : CompetitionType, optional
            Filter by competition type
            
        Returns
        -------
        list of dict
            List of competitions matching the filters
        """
        competitions = list(self.competitions.values())
        
        if status:
            competitions = [c for c in competitions if c['status'] == status]
        
        if competition_type:
            competitions = [c for c in competitions if c['type'] == competition_type]
        
        return competitions
    
    def register_participant(self, comp_id: str, user_id: str) -> bool:
        """
        Register a participant for a competition.
        
        Parameters
        ----------
        comp_id : str
            Competition ID
        user_id : str
            User ID
            
        Returns
        -------
        bool
            True if registration successful
        """
        comp = self.competitions.get(comp_id)
        if not comp:
            return False
        
        if user_id not in comp['participants']:
            comp['participants'].append(user_id)
            return True
        
        return False
    
    def start_competition(self, comp_id: str) -> bool:
        """
        Start a competition.
        
        Parameters
        ----------
        comp_id : str
            Competition ID
            
        Returns
        -------
        bool
            True if started successfully
        """
        comp = self.competitions.get(comp_id)
        if not comp:
            return False
        
        comp['status'] = CompetitionStatus.ACTIVE
        return True
    
    def end_competition(self, comp_id: str) -> bool:
        """
        End a competition.
        
        Parameters
        ----------
        comp_id : str
            Competition ID
            
        Returns
        -------
        bool
            True if ended successfully
        """
        comp = self.competitions.get(comp_id)
        if not comp:
            return False
        
        comp['status'] = CompetitionStatus.COMPLETED
        return True
    
    def get_active_competitions(self) -> List[Dict[str, Any]]:
        """
        Get all active competitions.
        
        Returns
        -------
        list of dict
            List of active competitions
        """
        return self.list_competitions(status=CompetitionStatus.ACTIVE)

"""
Strategy Submission - Handle strategy submissions for competitions
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import hashlib


class SubmissionStatus(Enum):
    """Status of a strategy submission"""
    PENDING = "pending"
    VALIDATING = "validating"
    VALID = "valid"
    INVALID = "invalid"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StrategySubmission:
    """
    Handle strategy submissions for competitions.
    
    Provides code validation, sandboxing, resource limits,
    and anti-cheating measures.
    
    Parameters
    ----------
    competition_id : str
        ID of the competition
    user_id : str
        ID of the submitting user
    strategy_code : str
        Strategy code to submit
    metadata : dict, optional
        Additional metadata about the submission
        
    Examples
    --------
    >>> submission = StrategySubmission(
    ...     competition_id="comp_123",
    ...     user_id="user_456",
    ...     strategy_code="def initialize(context): pass"
    ... )
    >>> is_valid = submission.validate()
    """
    
    def __init__(
        self,
        competition_id: str,
        user_id: str,
        strategy_code: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.competition_id = competition_id
        self.user_id = user_id
        self.strategy_code = strategy_code
        self.metadata = metadata or {}
        self.submission_id = self._generate_id()
        self.status = SubmissionStatus.PENDING
        self.created_at = datetime.now()
        self.validation_errors: List[str] = []
        self.resource_usage: Dict[str, float] = {}
    
    def _generate_id(self) -> str:
        """Generate a unique submission ID."""
        content = f"{self.competition_id}_{self.user_id}_{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def validate(self) -> bool:
        """
        Validate the submitted strategy code.
        
        Checks for:
        - Valid Python syntax
        - Required functions (initialize, handle_data)
        - No prohibited imports or operations
        - No lookahead bias indicators
        
        Returns
        -------
        bool
            True if validation passes
        """
        self.status = SubmissionStatus.VALIDATING
        self.validation_errors = []
        
        # Check for empty code
        if not self.strategy_code.strip():
            self.validation_errors.append("Strategy code is empty")
            self.status = SubmissionStatus.INVALID
            return False
        
        # Check for syntax errors
        try:
            compile(self.strategy_code, '<string>', 'exec')
        except SyntaxError as e:
            self.validation_errors.append(f"Syntax error: {e}")
            self.status = SubmissionStatus.INVALID
            return False
        
        # Check for required functions
        required_functions = ['initialize', 'handle_data']
        for func in required_functions:
            if f"def {func}" not in self.strategy_code:
                self.validation_errors.append(f"Missing required function: {func}")
        
        # Check for prohibited operations
        prohibited = [
            'import os',
            'import subprocess',
            'import sys',
            '__import__',
            'eval(',
            'exec(',
        ]
        for prohibited_str in prohibited:
            if prohibited_str in self.strategy_code:
                self.validation_errors.append(
                    f"Prohibited operation detected: {prohibited_str}"
                )
        
        # Check for lookahead bias indicators
        lookahead_indicators = [
            '.shift(-',  # Future data access
            'future',
            'tomorrow',
        ]
        for indicator in lookahead_indicators:
            if indicator in self.strategy_code.lower():
                self.validation_errors.append(
                    f"Potential lookahead bias detected: {indicator}"
                )
        
        if self.validation_errors:
            self.status = SubmissionStatus.INVALID
            return False
        
        self.status = SubmissionStatus.VALID
        return True
    
    def get_code_hash(self) -> str:
        """
        Get hash of the strategy code.
        
        Returns
        -------
        str
            SHA256 hash of the code
        """
        return hashlib.sha256(self.strategy_code.encode()).hexdigest()
    
    def set_resource_limits(
        self,
        max_cpu_time: float = 3600.0,  # 1 hour
        max_memory_mb: float = 2048.0,  # 2 GB
        max_execution_time: float = 7200.0,  # 2 hours
    ) -> Dict[str, float]:
        """
        Set resource limits for strategy execution.
        
        Parameters
        ----------
        max_cpu_time : float
            Maximum CPU time in seconds
        max_memory_mb : float
            Maximum memory in MB
        max_execution_time : float
            Maximum wall-clock execution time in seconds
            
        Returns
        -------
        dict
            Resource limits
        """
        limits = {
            'max_cpu_time': max_cpu_time,
            'max_memory_mb': max_memory_mb,
            'max_execution_time': max_execution_time,
        }
        self.metadata['resource_limits'] = limits
        return limits
    
    def record_resource_usage(self, usage: Dict[str, float]) -> None:
        """
        Record actual resource usage.
        
        Parameters
        ----------
        usage : dict
            Resource usage metrics
        """
        self.resource_usage = usage
    
    def check_duplicate(self, existing_submissions: List['StrategySubmission']) -> bool:
        """
        Check if this is a duplicate submission.
        
        Parameters
        ----------
        existing_submissions : list of StrategySubmission
            Previous submissions to check against
            
        Returns
        -------
        bool
            True if duplicate found
        """
        my_hash = self.get_code_hash()
        for submission in existing_submissions:
            if submission.get_code_hash() == my_hash:
                return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert submission to dictionary.
        
        Returns
        -------
        dict
            Submission data
        """
        return {
            'submission_id': self.submission_id,
            'competition_id': self.competition_id,
            'user_id': self.user_id,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'code_hash': self.get_code_hash(),
            'validation_errors': self.validation_errors,
            'resource_usage': self.resource_usage,
            'metadata': self.metadata,
        }


class SubmissionHistory:
    """
    Track submission history for a user in a competition.
    
    Parameters
    ----------
    competition_id : str
        Competition ID
    user_id : str
        User ID
    """
    
    def __init__(self, competition_id: str, user_id: str):
        self.competition_id = competition_id
        self.user_id = user_id
        self.submissions: List[StrategySubmission] = []
    
    def add_submission(self, submission: StrategySubmission) -> None:
        """Add a submission to history."""
        self.submissions.append(submission)
    
    def get_latest_submission(self) -> Optional[StrategySubmission]:
        """Get the most recent submission."""
        if not self.submissions:
            return None
        return self.submissions[-1]
    
    def get_valid_submissions(self) -> List[StrategySubmission]:
        """Get all valid submissions."""
        return [
            s for s in self.submissions
            if s.status == SubmissionStatus.VALID
        ]
    
    def count_submissions(self) -> int:
        """Get total number of submissions."""
        return len(self.submissions)

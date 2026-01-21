"""User data models"""
from typing import Dict, Any


class User:
    """User account model."""
    def __init__(self, user_id: str, email: str):
        self.user_id = user_id
        self.email = email


class UserProfile:
    """Extended user profile."""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.display_name = ""


class UserSettings:
    """User preferences."""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.settings: Dict[str, Any] = {}


class UserStats:
    """User trading statistics."""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.total_trades = 0
        self.win_rate = 0.0

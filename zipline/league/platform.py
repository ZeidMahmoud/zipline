"""Paper Trading League Platform"""
from typing import Dict, List
from enum import Enum


class Division(Enum):
    """League divisions"""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"


class PaperTradingLeague:
    """Gamified paper trading league."""
    
    def __init__(self, name: str):
        self.name = name
        self.players: Dict[str, dict] = {}
        self.current_season = 1
    
    def register_player(self, user_id: str) -> bool:
        """Register a player."""
        if user_id not in self.players:
            self.players[user_id] = {
                'division': Division.BRONZE,
                'xp': 0,
                'level': 1,
            }
            return True
        return False

"""Social Trading Platform"""
from typing import Dict, List


class SocialTradingPlatform:
    """Main social trading platform."""
    
    def __init__(self, name: str):
        self.name = name
        self.users: Dict[str, dict] = {}
        self.followers: Dict[str, List[str]] = {}
    
    def follow_user(self, follower_id: str, leader_id: str) -> bool:
        """Follow a user."""
        if leader_id not in self.followers:
            self.followers[leader_id] = []
        if follower_id not in self.followers[leader_id]:
            self.followers[leader_id].append(follower_id)
            return True
        return False

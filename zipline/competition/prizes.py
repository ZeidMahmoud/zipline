"""
Prize System - Manage competition prizes and distribution
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class PrizeType(Enum):
    """Types of prizes"""
    CASH = "cash"
    VIRTUAL_CURRENCY = "virtual_currency"
    ACHIEVEMENT = "achievement"
    FEATURE_UNLOCK = "feature_unlock"
    BADGE = "badge"


class PrizeTier(Enum):
    """Prize tiers"""
    FIRST_PLACE = "1st"
    SECOND_PLACE = "2nd"
    THIRD_PLACE = "3rd"
    TOP_10_PERCENT = "top_10_percent"
    TOP_25_PERCENT = "top_25_percent"
    PARTICIPANT = "participant"


class PrizePool:
    """
    Manage competition prize pools and distribution.
    
    Handles automatic prize distribution, prize tiers, virtual currency,
    and achievement unlocks.
    
    Parameters
    ----------
    competition_id : str
        ID of the competition
    total_prize_amount : float, optional
        Total prize pool amount
    currency : str, optional
        Currency type (default: 'USD')
        
    Examples
    --------
    >>> prize_pool = PrizePool("comp_123", total_prize_amount=10000.0)
    >>> prize_pool.set_tier_distribution({
    ...     PrizeTier.FIRST_PLACE: 0.5,
    ...     PrizeTier.SECOND_PLACE: 0.3,
    ...     PrizeTier.THIRD_PLACE: 0.2,
    ... })
    >>> winners = prize_pool.distribute_prizes(rankings)
    """
    
    def __init__(
        self,
        competition_id: str,
        total_prize_amount: float = 0.0,
        currency: str = 'USD',
    ):
        self.competition_id = competition_id
        self.total_prize_amount = total_prize_amount
        self.currency = currency
        self.tier_distribution: Dict[PrizeTier, float] = {}
        self.prize_awards: List[Dict[str, Any]] = []
        self.achievement_unlocks: Dict[str, List[str]] = {}
    
    def set_tier_distribution(
        self,
        distribution: Dict[PrizeTier, float],
    ) -> None:
        """
        Set prize distribution across tiers.
        
        Parameters
        ----------
        distribution : dict
            Mapping of tier to percentage (0-1) of total prize pool
        """
        # Validate distribution sums to 1.0
        total = sum(distribution.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Distribution must sum to 1.0, got {total}"
            )
        
        self.tier_distribution = distribution
    
    def calculate_prize_amounts(self) -> Dict[PrizeTier, float]:
        """
        Calculate prize amounts for each tier.
        
        Returns
        -------
        dict
            Mapping of tier to prize amount
        """
        return {
            tier: self.total_prize_amount * percentage
            for tier, percentage in self.tier_distribution.items()
        }
    
    def distribute_prizes(
        self,
        rankings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Distribute prizes based on rankings.
        
        Parameters
        ----------
        rankings : list of dict
            Ranked list of participants
            
        Returns
        -------
        list of dict
            Prize awards for each participant
        """
        awards = []
        prize_amounts = self.calculate_prize_amounts()
        total_participants = len(rankings)
        
        for i, entry in enumerate(rankings):
            user_id = entry['user_id']
            rank = entry['rank']
            
            # Determine tier
            tier = self._determine_tier(rank, total_participants)
            
            # Calculate prize
            prize_amount = 0.0
            if tier in prize_amounts:
                if tier in [PrizeTier.FIRST_PLACE, PrizeTier.SECOND_PLACE, PrizeTier.THIRD_PLACE]:
                    prize_amount = prize_amounts[tier]
                else:
                    # Split tier prize among all in that tier
                    num_in_tier = self._count_in_tier(total_participants, tier)
                    prize_amount = prize_amounts[tier] / num_in_tier
            
            # Create award
            award = {
                'user_id': user_id,
                'rank': rank,
                'tier': tier.value,
                'prize_amount': prize_amount,
                'currency': self.currency,
                'awarded_at': datetime.now().isoformat(),
                'competition_id': self.competition_id,
            }
            
            awards.append(award)
            self.prize_awards.append(award)
        
        return awards
    
    def _determine_tier(self, rank: int, total: int) -> PrizeTier:
        """Determine prize tier based on rank."""
        if rank == 1:
            return PrizeTier.FIRST_PLACE
        elif rank == 2:
            return PrizeTier.SECOND_PLACE
        elif rank == 3:
            return PrizeTier.THIRD_PLACE
        elif rank <= total * 0.1:
            return PrizeTier.TOP_10_PERCENT
        elif rank <= total * 0.25:
            return PrizeTier.TOP_25_PERCENT
        else:
            return PrizeTier.PARTICIPANT
    
    def _count_in_tier(self, total: int, tier: PrizeTier) -> int:
        """Count number of participants in a tier."""
        if tier == PrizeTier.TOP_10_PERCENT:
            return max(1, int(total * 0.1) - 3)  # Exclude top 3
        elif tier == PrizeTier.TOP_25_PERCENT:
            return max(1, int(total * 0.25) - int(total * 0.1))
        elif tier == PrizeTier.PARTICIPANT:
            return max(1, total - int(total * 0.25))
        return 1
    
    def award_virtual_currency(
        self,
        user_id: str,
        amount: float,
        reason: str = "",
    ) -> Dict[str, Any]:
        """
        Award virtual currency to a user.
        
        Parameters
        ----------
        user_id : str
            User ID
        amount : float
            Amount of virtual currency
        reason : str, optional
            Reason for the award
            
        Returns
        -------
        dict
            Award details
        """
        award = {
            'user_id': user_id,
            'type': PrizeType.VIRTUAL_CURRENCY.value,
            'amount': amount,
            'reason': reason,
            'awarded_at': datetime.now().isoformat(),
            'competition_id': self.competition_id,
        }
        
        self.prize_awards.append(award)
        return award
    
    def unlock_achievement(
        self,
        user_id: str,
        achievement: str,
    ) -> None:
        """
        Unlock an achievement for a user.
        
        Parameters
        ----------
        user_id : str
            User ID
        achievement : str
            Achievement identifier
        """
        if user_id not in self.achievement_unlocks:
            self.achievement_unlocks[user_id] = []
        
        if achievement not in self.achievement_unlocks[user_id]:
            self.achievement_unlocks[user_id].append(achievement)
            
            # Record as prize award
            self.prize_awards.append({
                'user_id': user_id,
                'type': PrizeType.ACHIEVEMENT.value,
                'achievement': achievement,
                'awarded_at': datetime.now().isoformat(),
                'competition_id': self.competition_id,
            })
    
    def get_user_awards(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all awards for a user.
        
        Parameters
        ----------
        user_id : str
            User ID
            
        Returns
        -------
        list of dict
            All awards for the user
        """
        return [
            award for award in self.prize_awards
            if award['user_id'] == user_id
        ]
    
    def get_total_awarded(self) -> float:
        """
        Get total amount awarded.
        
        Returns
        -------
        float
            Total amount of prizes awarded
        """
        return sum(
            award.get('prize_amount', 0.0)
            for award in self.prize_awards
            if 'prize_amount' in award
        )

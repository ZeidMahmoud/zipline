"""
Certification Levels

Defines certification levels and requirements.
"""

from typing import List, Dict

CERTIFICATION_LEVELS = [
    {
        'level': 1,
        'name': 'Zipline Certified Trader - Foundation',
        'requirements': ['complete_beginner_track', 'pass_foundation_exam'],
        'badge': 'bronze',
        'description': 'Master trading fundamentals and basic concepts'
    },
    {
        'level': 2,
        'name': 'Zipline Certified Algorithmic Trader',
        'requirements': ['level_1', 'complete_intermediate_track', 'pass_algo_exam', 'submit_strategy'],
        'badge': 'silver',
        'description': 'Build and backtest algorithmic trading strategies'
    },
    {
        'level': 3,
        'name': 'Zipline Certified Quant',
        'requirements': ['level_2', 'complete_advanced_track', 'pass_quant_exam', 'profitable_backtest'],
        'badge': 'gold',
        'description': 'Advanced quantitative finance and strategy development'
    },
    {
        'level': 4,
        'name': 'Zipline Master Quant',
        'requirements': ['level_3', 'complete_expert_track', 'pass_master_exam', 'publish_strategy'],
        'badge': 'platinum',
        'description': 'Expert-level trading systems and institutional strategies'
    },
    {
        'level': 5,
        'name': 'Zipline Trading Grandmaster',
        'requirements': ['level_4', 'win_competition', 'mentor_students', 'contribute_code'],
        'badge': 'diamond',
        'description': 'Elite mastery with community contributions and competition wins'
    }
]


def get_certification_level(level: int) -> Dict:
    """
    Get certification level details
    
    Args:
        level: Certification level (1-5)
        
    Returns:
        Certification level information
    """
    for cert in CERTIFICATION_LEVELS:
        if cert['level'] == level:
            return cert
    return {}


def list_certification_levels() -> List[Dict]:
    """
    List all certification levels
    
    Returns:
        List of certification levels
    """
    return CERTIFICATION_LEVELS


def get_next_level(current_level: int) -> Dict:
    """
    Get next certification level
    
    Args:
        current_level: Current certification level
        
    Returns:
        Next level information
    """
    next_level = current_level + 1
    return get_certification_level(next_level)


def check_requirements(user_achievements: List[str], target_level: int) -> Dict:
    """
    Check if user meets requirements for certification level
    
    Args:
        user_achievements: List of user's achievements
        target_level: Target certification level
        
    Returns:
        Dictionary with requirement status
    """
    cert = get_certification_level(target_level)
    
    if not cert:
        return {'eligible': False, 'reason': 'Invalid level'}
    
    requirements = cert['requirements']
    met_requirements = []
    missing_requirements = []
    
    for req in requirements:
        if req in user_achievements or req.startswith('level_') and check_level_requirement(user_achievements, req):
            met_requirements.append(req)
        else:
            missing_requirements.append(req)
    
    return {
        'eligible': len(missing_requirements) == 0,
        'met': met_requirements,
        'missing': missing_requirements,
        'progress': len(met_requirements) / len(requirements) * 100
    }


def check_level_requirement(achievements: List[str], level_req: str) -> bool:
    """Check if user has completed previous level"""
    # Extract level number from requirement like 'level_1'
    try:
        required_level = int(level_req.split('_')[1])
        # Check if user has certification for that level
        for achievement in achievements:
            if f'certified_level_{required_level}' in achievement:
                return True
    except (IndexError, ValueError):
        pass
    return False

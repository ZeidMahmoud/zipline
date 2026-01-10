"""
Learning Tracks

Predefined learning paths with courses and certifications.
"""

from typing import Dict, List

# Learning track definitions
LEARNING_TRACKS = {
    'beginner': {
        'name': 'Trading Fundamentals',
        'courses': [
            'intro_to_trading',
            'market_basics',
            'technical_analysis_101',
            'first_algorithm',
        ],
        'duration': '4 weeks',
        'certificate': 'Zipline Certified Trader - Foundation',
        'description': 'Start your trading journey with fundamental concepts'
    },
    'intermediate': {
        'name': 'Algorithmic Trading',
        'courses': [
            'python_for_trading',
            'backtesting_mastery',
            'strategy_development',
            'risk_management',
            'portfolio_optimization',
        ],
        'duration': '8 weeks',
        'certificate': 'Zipline Certified Algorithmic Trader',
        'description': 'Learn to build and test algorithmic trading strategies'
    },
    'advanced': {
        'name': 'Quantitative Finance',
        'courses': [
            'statistical_modeling',
            'machine_learning_trading',
            'options_strategies',
            'high_frequency_trading',
            'market_microstructure',
        ],
        'duration': '12 weeks',
        'certificate': 'Zipline Certified Quant',
        'description': 'Advanced quantitative techniques for professional trading'
    },
    'expert': {
        'name': 'Professional Trading Systems',
        'courses': [
            'institutional_strategies',
            'execution_algorithms',
            'alternative_data',
            'deep_learning_finance',
            'system_architecture',
        ],
        'duration': '16 weeks',
        'certificate': 'Zipline Master Quant',
        'description': 'Master-level strategies and system design'
    },
    'blockchain': {
        'name': 'DeFi & Blockchain Trading',
        'courses': [
            'blockchain_fundamentals',
            'defi_protocols',
            'smart_contract_trading',
            'mev_strategies',
            'cross_chain_arbitrage',
        ],
        'duration': '8 weeks',
        'certificate': 'Zipline Certified DeFi Trader',
        'description': 'Trade on blockchain and DeFi protocols'
    }
}


def get_learning_track(track_id: str) -> Dict:
    """
    Get learning track by ID
    
    Args:
        track_id: Track identifier
        
    Returns:
        Track information
    """
    return LEARNING_TRACKS.get(track_id, {})


def list_learning_tracks() -> List[Dict]:
    """
    List all learning tracks
    
    Returns:
        List of tracks with metadata
    """
    tracks = []
    for track_id, track_data in LEARNING_TRACKS.items():
        track_info = {
            'id': track_id,
            **track_data
        }
        tracks.append(track_info)
    return tracks


def get_recommended_track(experience_level: str) -> Dict:
    """
    Get recommended track based on experience
    
    Args:
        experience_level: User's experience level
        
    Returns:
        Recommended track
    """
    level_mapping = {
        'none': 'beginner',
        'beginner': 'beginner',
        'some': 'intermediate',
        'intermediate': 'intermediate',
        'advanced': 'advanced',
        'expert': 'expert',
        'professional': 'expert'
    }
    
    track_id = level_mapping.get(experience_level.lower(), 'beginner')
    return {'id': track_id, **LEARNING_TRACKS[track_id]}

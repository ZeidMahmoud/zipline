"""
Quantitative Finance Module for Zipline.

This module provides advanced quantitative methods including statistics,
stochastic calculus, Bayesian inference, portfolio optimization, market
microstructure analysis, and signal processing.
"""

try:
    from . import statistics
    from . import stochastic
    from . import bayesian
    from . import optimization
    from . import microstructure
    from . import signals
    
    __all__ = [
        'statistics',
        'stochastic',
        'bayesian',
        'optimization',
        'microstructure',
        'signals',
    ]
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Some quant modules not available: {e}")
    __all__ = []

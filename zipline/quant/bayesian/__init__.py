"""Bayesian Inference for Finance."""
try:
    from .portfolio import BayesianPortfolio
    from .regime import BayesianRegimeSwitching
    from .inference import MCMCSampler, VariationalInference, BayesianRegression
    from .sharpe import ProbabilisticSharpe
    __all__ = ['BayesianPortfolio', 'BayesianRegimeSwitching', 'MCMCSampler', 
               'VariationalInference', 'BayesianRegression', 'ProbabilisticSharpe']
except ImportError:
    __all__ = []

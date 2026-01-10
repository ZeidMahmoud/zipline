"""
Machine Learning integration for Zipline Pipeline API.

This module provides ML-based factors, model wrappers, and feature
engineering capabilities for use in Zipline trading algorithms.
"""
from .factors import MLPredictionFactor, FeatureUnion, RollingRegression
from .models import SklearnModelWrapper, ModelRegistry
from .features import TechnicalFeatures

__all__ = [
    'MLPredictionFactor',
    'FeatureUnion',
    'RollingRegression',
    'SklearnModelWrapper',
    'ModelRegistry',
    'TechnicalFeatures',
]

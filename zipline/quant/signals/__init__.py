"""Signal Processing for Trading."""
try:
    from .fourier import FFTAnalyzer
    from .wavelet import WaveletDecomposer, WaveletDenoiser
    from .filters import KalmanTrendFilter, HodrickPrescottFilter
    from .technical import AdaptiveMovingAverage, EhlerIndicators
    __all__ = ['FFTAnalyzer', 'WaveletDecomposer', 'WaveletDenoiser',
               'KalmanTrendFilter', 'HodrickPrescottFilter',
               'AdaptiveMovingAverage', 'EhlerIndicators']
except ImportError:
    __all__ = []

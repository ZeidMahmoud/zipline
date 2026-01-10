"""Wavelet Analysis."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class WaveletDecomposer:
    """Multi-scale wavelet decomposition."""
    
    def __init__(self, wavelet='db4', level=3):
        self.wavelet = wavelet
        self.level = level
    
    def decompose(self, signal):
        """Decompose signal using wavelets."""
        try:
            import pywt
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
            return coeffs
        except ImportError:
            logger.warning("pywavelets required for wavelet analysis")
            return [signal]

class WaveletDenoiser:
    """Wavelet-based denoising."""
    
    def __init__(self, wavelet='db4', threshold='soft'):
        self.wavelet = wavelet
        self.threshold = threshold
    
    def denoise(self, signal):
        """Denoise signal using wavelets."""
        try:
            import pywt
            coeffs = pywt.wavedec(signal, self.wavelet)
            
            # Threshold wavelet coefficients
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))
            
            coeffs_thresh = [pywt.threshold(c, threshold, mode=self.threshold) for c in coeffs]
            
            return pywt.waverec(coeffs_thresh, self.wavelet)
        except ImportError:
            logger.warning("pywavelets required")
            return signal

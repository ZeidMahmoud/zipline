"""Fourier Analysis for Cycle Detection."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class FFTAnalyzer:
    """FFT-based cycle detection."""
    
    def __init__(self):
        pass
    
    def analyze(self, signal):
        """Perform FFT analysis."""
        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal))
        power_spectrum = np.abs(fft_result)**2
        
        return {
            'frequencies': frequencies,
            'power_spectrum': power_spectrum,
            'dominant_frequency': frequencies[np.argmax(power_spectrum[1:]) + 1]
        }
    
    def filter_frequencies(self, signal, cutoff_freq):
        """Filter signal by frequency."""
        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal))
        
        # Apply frequency filter
        fft_result[np.abs(frequencies) > cutoff_freq] = 0
        
        return np.fft.ifft(fft_result).real

"""GPU Acceleration Utilities."""
import logging
logger = logging.getLogger(__name__)

def cuda_available():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_device():
    """Get compute device (cuda or cpu)."""
    if cuda_available():
        return 'cuda'
    return 'cpu'

logger.info(f"GPU available: {cuda_available()}")

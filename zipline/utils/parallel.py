"""Parallel Processing Utilities."""
import multiprocessing
import logging
logger = logging.getLogger(__name__)

def get_n_jobs(n_jobs=None):
    """Get number of parallel jobs."""
    if n_jobs is None:
        return multiprocessing.cpu_count()
    elif n_jobs == -1:
        return multiprocessing.cpu_count()
    else:
        return min(n_jobs, multiprocessing.cpu_count())

def parallel_apply(func, items, n_jobs=None):
    """Apply function to items in parallel."""
    n_jobs = get_n_jobs(n_jobs)
    
    if n_jobs == 1:
        return [func(item) for item in items]
    
    with multiprocessing.Pool(n_jobs) as pool:
        return pool.map(func, items)

logger.info(f"Available CPUs: {multiprocessing.cpu_count()}")

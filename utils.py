# Terminal colors
import hashlib
from collections import deque


def compute_file_md5(file_path: str) -> str:
    """Compute MD5 hash of a file's contents."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Regular colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    
    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

class AccuracyTracker:
    """
    Tracks accuracy using a sliding window for smoothed threshold checks.
    
    This provides a low-pass filter on accuracy measurements to avoid
    triggering phase transitions on noisy single-point measurements.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of measurements to average over.
        """
        self.window_size = window_size
        self._window: deque[float] = deque(maxlen=window_size)
    
    def update(self, accuracy: float) -> None:
        """Add a new accuracy measurement."""
        self._window.append(accuracy)
    
    @property
    def latest(self) -> float:
        """Get the most recent raw accuracy measurement."""
        if not self._window:
            return 0.0
        return self._window[-1]
    
    @property
    def smoothed(self) -> float:
        """Get the smoothed (windowed average) accuracy."""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)
    
    @property
    def is_stable(self) -> bool:
        """Check if we have enough samples for a stable estimate."""
        return len(self._window) >= self.window_size
    
    def above_threshold(self, threshold: float) -> bool:
        """
        Check if smoothed accuracy is above threshold.
        
        Args:
            threshold: Threshold in (0-1).
            
        Returns:
            True if smoothed accuracy >= threshold AND window is stable.
        """
        return self.is_stable and self.smoothed >= threshold
    
    def reset(self) -> None:
        """Clear all measurements."""
        self._window.clear()



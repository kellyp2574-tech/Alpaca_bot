"""Thread-safe rate limiter for Alpaca API calls.

Provides a RateLimitedSession that wraps requests.Session with a shared
sliding-window rate limiter. All sessions sharing the same limiter instance
count against a single budget (e.g., 100 requests/minute across Trading + Data APIs).
"""
import threading
import time
import logging
from collections import deque
import requests

logger = logging.getLogger(__name__)


class SlidingWindowLimiter:
    """Thread-safe sliding-window rate limiter.
    
    Tracks timestamps of recent requests in a deque. Before each request,
    evicts expired entries and blocks if the window is full.
    """

    def __init__(self, max_calls: int = 100, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._timestamps: deque = deque()
        self._lock = threading.Lock()
        self._total_calls: int = 0  # lifetime counter for diagnostics

    def acquire(self):
        """Block until a request slot is available, then record the timestamp."""
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window_seconds

                # Evict expired timestamps
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()

                in_window = len(self._timestamps)
                if in_window < self.max_calls:
                    self._timestamps.append(now)
                    self._total_calls += 1
                    # Early warning at 70% utilization
                    if in_window >= int(self.max_calls * 0.7):
                        logger.warning(
                            f"Rate limiter: high usage {in_window + 1}/{self.max_calls} "
                            f"in last {self.window_seconds:.0f}s"
                        )
                    return

                # Window is full — calculate wait time until the oldest entry expires
                wait = self._timestamps[0] - cutoff
                in_window = len(self._timestamps)

            logger.warning(
                f"Rate limiter: {in_window}/{self.max_calls} calls in last "
                f"{self.window_seconds:.0f}s — throttling {wait:.1f}s"
            )
            time.sleep(wait)


class RateLimitedSession(requests.Session):
    """requests.Session subclass that enforces a shared rate limit.
    
    Drop-in replacement for requests.Session. Pass a SlidingWindowLimiter
    instance so multiple sessions can share the same budget.
    """

    def __init__(self, limiter: SlidingWindowLimiter):
        super().__init__()
        self._limiter = limiter

    def request(self, method, url, **kwargs):
        self._limiter.acquire()
        return super().request(method, url, **kwargs)


# Module-level shared limiter: 80 requests per 60 seconds across all Alpaca sessions
# (20-call buffer under Alpaca's 100/min hard limit for retries and bursts)
_alpaca_limiter = SlidingWindowLimiter(max_calls=80, window_seconds=60.0)


def create_alpaca_session() -> RateLimitedSession:
    """Create a rate-limited session that shares the global Alpaca budget."""
    return RateLimitedSession(_alpaca_limiter)


def get_api_call_count() -> int:
    """Return total lifetime API calls through the shared limiter."""
    return _alpaca_limiter._total_calls


def get_calls_in_window() -> int:
    """Return how many calls are currently in the sliding window."""
    with _alpaca_limiter._lock:
        now = time.monotonic()
        cutoff = now - _alpaca_limiter.window_seconds
        while _alpaca_limiter._timestamps and _alpaca_limiter._timestamps[0] <= cutoff:
            _alpaca_limiter._timestamps.popleft()
        return len(_alpaca_limiter._timestamps)

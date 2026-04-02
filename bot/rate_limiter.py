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

    def acquire(self):
        """Block until a request slot is available, then record the timestamp."""
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window_seconds

                # Evict expired timestamps
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return

                # Window is full — calculate wait time until the oldest entry expires
                wait = self._timestamps[0] - cutoff

            logger.debug(f"Rate limiter: {self.max_calls}/{self.window_seconds}s limit reached, waiting {wait:.1f}s")
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


# Module-level shared limiter: 100 requests per 60 seconds across all Alpaca sessions
_alpaca_limiter = SlidingWindowLimiter(max_calls=100, window_seconds=60.0)


def create_alpaca_session() -> RateLimitedSession:
    """Create a rate-limited session that shares the global Alpaca budget."""
    return RateLimitedSession(_alpaca_limiter)

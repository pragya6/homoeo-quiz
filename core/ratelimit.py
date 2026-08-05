"""Client-side throttle and retry.

Two separate problems, solved separately:

  TokenBucket  - proactive. Never *send* more than N requests/minute.
  with_retry   - reactive. If the server 429s anyway, back off and try again.

You need both. The bucket handles our own traffic; retry handles the fact that
free-tier quotas are shared per-project and can be consumed elsewhere.
"""

import random
import threading
import time
from typing import Callable, TypeVar

from config import BACKOFF_BASE_SECONDS, MAX_RETRIES

T = TypeVar("T")


class TokenBucket:
    """Classic leaky bucket. Thread-safe because Gradio serves concurrently."""

    def __init__(self, rpm: int):
        self.capacity = rpm
        self.tokens = float(rpm)
        self.refill_per_sec = rpm / 60.0
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                self.tokens = min(
                    self.capacity, self.tokens + (now - self.updated) * self.refill_per_sec
                )
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.15)


_buckets: dict[str, TokenBucket] = {}
_buckets_lock = threading.Lock()


def bucket_for(model: str, rpm: int) -> TokenBucket:
    with _buckets_lock:
        if model not in _buckets:
            _buckets[model] = TokenBucket(rpm)
        return _buckets[model]


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in ("429", "resource_exhausted", "503", "unavailable", "deadline"))


def with_retry(fn: Callable[[], T]) -> T:
    """Exponential backoff with jitter. Jitter matters: without it, concurrent
    students retry in lockstep and re-trip the same limit."""
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - SDK raises a wide surface
            last = exc
            if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                raise
            sleep_s = BACKOFF_BASE_SECONDS * (2**attempt) + random.uniform(0, 0.6)
            time.sleep(sleep_s)
    raise last  # unreachable, keeps type checkers happy

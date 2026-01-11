"""
Session-based rate limiting for API protection.

Design decisions:
- Sliding window algorithm: Smoother than fixed windows, no burst at boundaries
- Streamlit session_state storage: No external dependencies, per-session isolation
- deque for O(1) operations: Efficient timestamp management

Limitations:
- Per-session, not per-user: Refresh browser = new quota
- In-memory: Lost on server restart
- Single-node: Not suitable for distributed deployment

For production, consider:
- Redis-based rate limiting (distributed, persistent)
- Token bucket for burst allowance
- IP-based limits (with proxy awareness)
- Graduated responses (warning → throttle → block)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, MutableMapping


def check_rate_limit(
    session_state: MutableMapping[str, Any],
    key: str,
    max_requests: int,
    window_seconds: int,
) -> tuple[bool, int]:
    """
    Check if request is within rate limit using sliding window.

    Args:
        session_state: Streamlit session state or similar mutable mapping
        key: Unique key for this rate limit bucket
        max_requests: Maximum requests allowed in window
        window_seconds: Window duration in seconds
        
    Returns:
        Tuple of (allowed, retry_after_seconds)
        
    Usage:
        allowed, retry = check_rate_limit(st.session_state, "chat", 20, 60)
        if not allowed:
            st.warning(f"Too many requests. Retry in {retry}s")
    """
    # Edge case: zero or negative max_requests always blocks
    if max_requests <= 0:
        return False, max(1, window_seconds)

    now = time.time()
    dq: Deque[float] | None = session_state.get(key)
    if dq is None:
        dq = deque()
        session_state[key] = dq

    # purge old entries outside window
    cutoff = now - window_seconds
    while dq and dq[0] < cutoff:
        dq.popleft()

    if len(dq) >= max_requests:
        retry_after = int(dq[0] + window_seconds - now) + 1
        return False, max(1, retry_after)

    dq.append(now)
    return True, 0

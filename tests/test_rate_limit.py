"""
Tests for session-based rate limiting.

Tests cover:
- Sliding window algorithm behavior
- Request counting and blocking
- Window expiration and reset
- Edge cases
"""

import time
from collections import deque

from backend.rate_limit import check_rate_limit


def test_allows_first_request():
    """Test that first request is always allowed."""
    session = {}

    allowed, retry_after = check_rate_limit(
        session, key="test", max_requests=5, window_seconds=60
    )

    assert allowed is True
    assert retry_after == 0


def test_allows_requests_under_limit():
    """Test that requests under the limit are allowed."""
    session = {}

    for i in range(5):
        allowed, _ = check_rate_limit(
            session, key="test", max_requests=5, window_seconds=60
        )
        assert allowed is True, f"Request {i+1} should be allowed"


def test_blocks_requests_over_limit():
    """Test that requests over the limit are blocked."""
    session = {}

    # Use up the quota
    for _ in range(5):
        check_rate_limit(session, key="test", max_requests=5, window_seconds=60)

    # Next request should be blocked
    allowed, retry_after = check_rate_limit(
        session, key="test", max_requests=5, window_seconds=60
    )

    assert allowed is False
    assert retry_after > 0


def test_retry_after_is_positive():
    """Test that retry_after is always a positive value when blocked."""
    session = {}

    # Exhaust quota
    for _ in range(3):
        check_rate_limit(session, key="test", max_requests=3, window_seconds=60)

    _, retry_after = check_rate_limit(
        session, key="test", max_requests=3, window_seconds=60
    )

    assert retry_after >= 1


def test_window_expiration_allows_new_requests():
    """Test that requests are allowed after window expires."""
    session = {}

    # Use up quota with a very short window
    for _ in range(2):
        check_rate_limit(session, key="test", max_requests=2, window_seconds=1)

    # Should be blocked now
    allowed, _ = check_rate_limit(
        session, key="test", max_requests=2, window_seconds=1
    )
    assert allowed is False

    # Wait for window to expire
    time.sleep(1.1)

    # Should be allowed again
    allowed, _ = check_rate_limit(
        session, key="test", max_requests=2, window_seconds=1
    )
    assert allowed is True


def test_sliding_window_behavior():
    """Test that old requests slide out of window."""
    session = {}
    window_seconds = 1

    # Make first request
    check_rate_limit(session, key="test", max_requests=2, window_seconds=window_seconds)

    # Wait half the window
    time.sleep(0.6)

    # Make second request
    check_rate_limit(session, key="test", max_requests=2, window_seconds=window_seconds)

    # Should be blocked (2 requests in window)
    allowed, _ = check_rate_limit(
        session, key="test", max_requests=2, window_seconds=window_seconds
    )
    assert allowed is False

    # Wait for first request to slide out
    time.sleep(0.5)

    # Now should be allowed (only 1 request in window)
    allowed, _ = check_rate_limit(
        session, key="test", max_requests=2, window_seconds=window_seconds
    )
    assert allowed is True


def test_independent_keys():
    """Test that different keys have independent rate limits."""
    session = {}

    # Exhaust quota for key "a"
    for _ in range(3):
        check_rate_limit(session, key="a", max_requests=3, window_seconds=60)

    # Key "a" should be blocked
    allowed_a, _ = check_rate_limit(
        session, key="a", max_requests=3, window_seconds=60
    )
    assert allowed_a is False

    # Key "b" should still be allowed
    allowed_b, _ = check_rate_limit(
        session, key="b", max_requests=3, window_seconds=60
    )
    assert allowed_b is True


def test_independent_sessions():
    """Test that different sessions have independent rate limits."""
    session1 = {}
    session2 = {}

    # Exhaust quota for session 1
    for _ in range(3):
        check_rate_limit(session1, key="test", max_requests=3, window_seconds=60)

    # Session 1 should be blocked
    allowed1, _ = check_rate_limit(
        session1, key="test", max_requests=3, window_seconds=60
    )
    assert allowed1 is False

    # Session 2 should still be allowed
    allowed2, _ = check_rate_limit(
        session2, key="test", max_requests=3, window_seconds=60
    )
    assert allowed2 is True


def test_creates_deque_on_first_request():
    """Test that the rate limiter creates a deque in session state."""
    session = {}

    check_rate_limit(session, key="my_limit", max_requests=5, window_seconds=60)

    assert "my_limit" in session
    assert isinstance(session["my_limit"], deque)


def test_zero_max_requests_blocks_all():
    """Test edge case where max_requests is 0."""
    session = {}

    allowed, _ = check_rate_limit(
        session, key="test", max_requests=0, window_seconds=60
    )

    assert allowed is False


def test_large_max_requests():
    """Test with a large max_requests value."""
    session = {}

    for _i in range(100):
        allowed, _ = check_rate_limit(
            session, key="test", max_requests=1000, window_seconds=60
        )
        assert allowed is True


def test_exact_limit_boundary():
    """Test behavior at exactly the limit."""
    session = {}

    # Make exactly max_requests requests
    for i in range(10):
        allowed, _ = check_rate_limit(
            session, key="test", max_requests=10, window_seconds=60
        )
        assert allowed is True, f"Request {i+1} of 10 should be allowed"

    # The 11th request should be blocked
    allowed, _ = check_rate_limit(
        session, key="test", max_requests=10, window_seconds=60
    )
    assert allowed is False

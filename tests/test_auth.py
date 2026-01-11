"""
Tests for authentication module.

Tests cover:
- Auth bypass when disabled
- Login flow (mocked streamlit components)
- Audit logging for auth events

Note: These tests mock streamlit components since they require browser context.
"""

from unittest.mock import MagicMock, patch

import pytest


class MockSessionState(dict):
    """Dict subclass that allows attribute assignment for mocking st.session_state."""
    pass


class StopExecution(Exception):
    """Exception to simulate st.stop() behavior."""
    pass


# ============================================================================
# AUTH DISABLED TESTS
# ============================================================================


def test_require_auth_returns_anonymous_when_disabled():
    """Test that require_auth returns 'anonymous' when auth is disabled."""
    with patch("backend.auth.settings") as mock_settings:
        mock_settings.auth_enabled = False

        from backend.auth import require_auth

        result = require_auth()

        assert result == "anonymous"


def test_render_logout_does_nothing_when_disabled():
    """Test that render_logout is a no-op when auth is disabled."""
    with patch("backend.auth.settings") as mock_settings:
        mock_settings.auth_enabled = False

        from backend.auth import render_logout

        # Should not raise
        render_logout()


# ============================================================================
# AUTH ENABLED TESTS (with mocked streamlit)
# ============================================================================


@pytest.fixture
def mock_streamlit():
    """Mock streamlit components for testing."""
    with patch("backend.auth.st") as mock_st:
        session_state = MockSessionState()
        mock_st.session_state = session_state
        mock_st.secrets = {
            "auth": {
                "cookie_name": "test_cookie",
                "cookie_key": "test_key_32_chars_for_testing!!",
                "cookie_expiry_days": 1,
                "credentials": {
                    "usernames": {
                        "testuser": {
                            "name": "Test User",
                            "password": "$2b$12$hashedpassword"
                        }
                    }
                }
            }
        }
        # Make st.stop() raise an exception to actually stop execution
        mock_st.stop.side_effect = StopExecution()
        yield mock_st


@pytest.fixture
def mock_settings_enabled():
    """Mock settings with auth enabled."""
    with patch("backend.auth.settings") as mock_settings:
        mock_settings.auth_enabled = True
        yield mock_settings


def test_get_authenticator_creates_instance(mock_streamlit):
    """Test that get_authenticator creates an Authenticate instance."""
    with patch("backend.auth.stauth.Authenticate") as MockAuth:
        MockAuth.return_value = MagicMock()

        from backend.auth import get_authenticator

        auth = get_authenticator()

        MockAuth.assert_called_once()
        assert auth is not None


def test_require_auth_stops_on_none_result(mock_streamlit, mock_settings_enabled):
    """Test that require_auth stops when login returns None."""
    with patch("backend.auth.stauth.Authenticate") as MockAuth:
        mock_auth = MagicMock()
        mock_auth.login.return_value = None  # New API returns None before login
        MockAuth.return_value = mock_auth

        from backend.auth import require_auth

        # Should call st.stop() which raises StopExecution
        with pytest.raises(StopExecution):
            require_auth()

        mock_streamlit.info.assert_called()


def test_require_auth_stops_on_none_status(mock_streamlit, mock_settings_enabled):
    """Test that require_auth stops when login status is None (not attempted)."""
    with patch("backend.auth.stauth.Authenticate") as MockAuth:
        mock_auth = MagicMock()
        mock_auth.login.return_value = (None, None, None)
        MockAuth.return_value = mock_auth

        from backend.auth import require_auth

        # Should call st.stop() which raises StopExecution
        with pytest.raises(StopExecution):
            require_auth()

        mock_streamlit.info.assert_called()


def test_require_auth_behavior_on_failed_login(mock_streamlit, mock_settings_enabled):
    """Test behavior when authentication_status is False (failed login)."""
    with patch("backend.auth.get_authenticator") as mock_get_auth:
        with patch("backend.auth.log_auth") as mock_log:
            mock_get_auth.return_value.login = MagicMock()

            # Simulate failed login state (set by authenticator)
            mock_streamlit.session_state["session_id"] = "test_session"
            mock_streamlit.session_state["authentication_status"] = False

            from backend.auth import require_auth

            with pytest.raises(StopExecution):
                require_auth()

            # Behavior: should show error and log failure
            mock_streamlit.error.assert_called()
            mock_log.assert_called_once()
            assert mock_log.call_args[0][2] == "login_failed"


def test_require_auth_behavior_on_successful_login(mock_streamlit, mock_settings_enabled):
    """Test behavior when authentication_status is True (successful login)."""
    with patch("backend.auth.get_authenticator") as mock_get_auth:
        with patch("backend.auth.log_auth") as mock_log:
            mock_get_auth.return_value.login = MagicMock()

            # Simulate successful login state
            mock_streamlit.session_state["session_id"] = "test_session"
            mock_streamlit.session_state["authentication_status"] = True
            mock_streamlit.session_state["username"] = "testuser"

            from backend.auth import require_auth

            result = require_auth()

            # Behavior: should return username and log success
            assert result == "testuser"
            mock_log.assert_called_once()
            assert mock_log.call_args[0][2] == "login_success"


def test_require_auth_skips_log_when_already_logged(mock_streamlit, mock_settings_enabled):
    """Test that repeated auth checks don't re-log (cookie restore scenario)."""
    with patch("backend.auth.get_authenticator") as mock_get_auth:
        with patch("backend.auth.log_auth") as mock_log:
            mock_get_auth.return_value.login = MagicMock()

            # Simulate already authenticated + already logged
            mock_streamlit.session_state["session_id"] = "test_session"
            mock_streamlit.session_state["authentication_status"] = True
            mock_streamlit.session_state["username"] = "testuser"
            mock_streamlit.session_state["auth_logged"] = True  # Already logged

            from backend.auth import require_auth

            result = require_auth()

            # Behavior: should return username but NOT log again
            assert result == "testuser"
            mock_log.assert_not_called()


# ============================================================================
# AUDIT LOG INTEGRATION
# ============================================================================


def test_log_auth_function_exists():
    """Test that log_auth function is importable."""
    from backend.audit_log import log_auth

    assert callable(log_auth)


def test_log_auth_accepts_required_params():
    """Test that log_auth accepts the expected parameters."""
    with patch("backend.audit_log._get_audit_logger") as mock_logger:
        mock_logger.return_value.info = MagicMock()

        from backend.audit_log import log_auth

        # Should not raise
        log_auth(
            request_id="req123",
            session_id="sess456",
            action="login_success",
            username="testuser",
        )


def test_log_auth_accepts_all_action_types():
    """Test that log_auth accepts all valid action types."""
    with patch("backend.audit_log._get_audit_logger") as mock_logger:
        mock_logger.return_value.info = MagicMock()

        from backend.audit_log import log_auth

        for action in ["login_success", "login_failed", "logout"]:
            log_auth(
                request_id="req",
                session_id="sess",
                action=action,
            )

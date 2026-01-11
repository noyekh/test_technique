"""
Authentication module using streamlit-authenticator.

Design: Simple secrets-based auth for PoC, replaceable with OAuth2/LDAP.

SECURITY:
- Passwords stored as bcrypt hashes in .streamlit/secrets.toml
- Cookie-based session persistence
- Audit logging for all auth events
"""

from __future__ import annotations

import streamlit as st
import streamlit_authenticator as stauth

from backend.audit_log import generate_request_id, log_auth
from backend.settings import settings


def get_authenticator() -> stauth.Authenticate:
    """Get or create authenticator, stored in session_state to persist across pages.

    See: https://github.com/mkhorasani/Streamlit-Authenticator/issues/156
    Storing in session_state prevents DuplicateWidgetId errors and preserves login state.
    """
    # Return cached authenticator if exists
    if "authenticator" in st.session_state:
        return st.session_state["authenticator"]

    config = st.secrets["auth"]

    # Deep copy credentials to avoid "Secrets does not support item assignment" error
    # streamlit-authenticator modifies credentials to track failed login attempts
    credentials = {"usernames": {}}
    for username, user_data in config["credentials"]["usernames"].items():
        credentials["usernames"][username] = {
            "email": user_data.get("email", ""),
            "name": user_data["name"],
            "password": user_data["password"],
        }

    authenticator = stauth.Authenticate(
        credentials=credentials,
        cookie_name=config["cookie_name"],
        cookie_key=config["cookie_key"],
        cookie_expiry_days=config["cookie_expiry_days"],
    )

    # Cache in session_state
    st.session_state["authenticator"] = authenticator
    return authenticator


def require_auth() -> str:
    """
    Require authentication. Returns username if authenticated.
    Displays login form and stops execution if not.
    """
    if not settings.auth_enabled:
        return "anonymous"

    authenticator = get_authenticator()

    # Call login() - this renders the form AND checks cookies
    authenticator.login(location="main")

    # Check authentication status from session_state (set by authenticator)
    status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")
    st.session_state.get("name")

    session_id = st.session_state.get("session_id", "unknown")
    request_id = generate_request_id()

    if status is False:
        log_auth(request_id, session_id, "login_failed")
        st.error("Nom d'utilisateur ou mot de passe incorrect")
        st.stop()
    elif status is None:
        st.info("Veuillez vous connecter pour accéder à l'application")
        st.stop()

    # Success - log only on fresh login (not cookie restore)
    if "auth_logged" not in st.session_state:
        log_auth(request_id, session_id, "login_success", username)
        st.session_state["auth_logged"] = True

    return username


def render_logout() -> None:
    """Render logout button in sidebar."""
    if not settings.auth_enabled:
        return
    authenticator = get_authenticator()
    authenticator.logout("Déconnexion", location="sidebar")

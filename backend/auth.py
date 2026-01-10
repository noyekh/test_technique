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
    """Create authenticator from Streamlit secrets."""
    config = st.secrets["auth"]
    return stauth.Authenticate(
        credentials={"usernames": dict(config["credentials"]["usernames"])},
        cookie_name=config["cookie_name"],
        cookie_key=config["cookie_key"],
        cookie_expiry_days=config["cookie_expiry_days"],
    )


def require_auth() -> str:
    """
    Require authentication. Returns username if authenticated.
    Displays login form and stops execution if not.
    """
    if not settings.auth_enabled:
        return "anonymous"

    authenticator = get_authenticator()
    name, status, username = authenticator.login(location="main")

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

    st.session_state["username"] = username
    return username


def render_logout() -> None:
    """Render logout button in sidebar."""
    if not settings.auth_enabled:
        return
    authenticator = get_authenticator()
    authenticator.logout("Déconnexion", location="sidebar")

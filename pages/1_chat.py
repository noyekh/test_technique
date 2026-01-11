"""
Chatbot page - RAG-based Q&A interface.

SECURITY:
- Streaming is DISABLED by default for legal contexts
- All responses are buffered and validated BEFORE display
- Audit logging with allowlist policy
"""

import logging
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from backend import db
from backend.audit_log import (
    RequestTimer,
    generate_request_id,
    log_error,
    log_query,
)
from backend.logging_config import setup_logging
from backend.rag import (
    answer_question_buffered,
    refusal,
)
from backend.auth import render_logout, require_auth
from backend.rate_limit import check_rate_limit
from backend.settings import settings

load_dotenv()
setup_logging()
db.init_db()

logger = logging.getLogger(__name__)


def _display_sources(sources: list[dict] | None) -> None:
    """Display sources in a user-friendly expander (no technical jargon)."""
    with st.expander("🔎 Sources citées"):
        if not sources:
            st.caption("Aucune source (refus ou documents insuffisants).")
        else:
            for s in sources:
                st.markdown(f"📄 **[Source {s['i']}]** — *{s['source']}*")

# Note: st.set_page_config() est dans main.py (st.navigation)
username = require_auth()  # Bloque si non authentifié
st.title("💬 Cabinet Emilia Parenti — Chatbot RAG")

# Welcome info (replaces main.py landing page)
with st.expander("ℹ️ À propos de ce PoC", expanded=False):
    st.markdown(
        """
**Chatbot strictement basé sur les documents uploadés.**

1. Va dans **Documents** pour uploader / supprimer et vectoriser
2. Reviens ici pour poser des questions

**Architecture v1.10** :
- ✅ Réponses validées avant affichage (pas de streaming)
- ✅ Citations vérifiées post-génération
- ✅ Logs minimalistes (allowlist RGPD)
- ✅ Prompt durci : documents = données non fiables
"""
    )

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY manquant : impossible d'appeler le modèle.")
    st.stop()

# Initialize session ID for audit
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex[:16]

# Initialize conversation
default_conv = db.ensure_default_conversation()
if "active_conv_id" not in st.session_state:
    st.session_state["active_conv_id"] = default_conv

# Sidebar: conversation management
with st.sidebar:
    st.header("Conversations")

    convs = db.list_conversations()
    options = [c["conv_id"] for c in convs] if convs else [default_conv]
    labels = {c["conv_id"]: c["title"] for c in convs}

    def _fmt(cid: str) -> str:
        return labels.get(cid, "Nouvelle conversation")

    current = st.session_state["active_conv_id"]
    if current not in options:
        current = options[0]
        st.session_state["active_conv_id"] = current

    selected = st.selectbox(
        "Choisir", options, index=options.index(current), format_func=_fmt
    )
    st.session_state["active_conv_id"] = selected

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ Nouvelle", use_container_width=True):
            new_id = db.new_conversation()
            st.session_state["active_conv_id"] = new_id
            st.rerun()
    with col_b:
        # Only show delete if more than one conversation exists
        if len(convs) > 1:
            if st.button("🗑️ Supprimer", use_container_width=True, key="delete_conv_btn"):
                db.delete_conversation(selected)
                # Switch to another conversation
                remaining = [c for c in convs if c["conv_id"] != selected]
                st.session_state["active_conv_id"] = remaining[0]["conv_id"] if remaining else db.ensure_default_conversation()
                st.rerun()

    st.divider()

    render_logout()
    st.caption(f"Connecté: {username}")

    # Technical details in expander (for developers)
    with st.expander("⚙️ Paramètres techniques"):
        st.caption(f"Rate limit: {settings.rate_limit_max_requests} req / {settings.rate_limit_window_seconds}s")
        st.caption(f"Modèle: {settings.openai_chat_model}")
        st.caption("Mode sécurisé: streaming désactivé, réponses validées")

        # Vectorstore diagnostic (moved from main area)
        st.divider()
        st.caption("**Vectorstore:**")
        try:
            from backend.rag_runtime import vectorstore
            vs = vectorstore()
            collection = vs.get()
            doc_count = len(collection.get("ids", []))
            if doc_count > 0:
                st.caption(f"✅ {doc_count} chunks indexés")
            else:
                st.caption("⚠️ Vide — uploadez des documents")
        except Exception as e:
            st.caption(f"❌ Erreur: {e}")
        if st.button("🔄 Vider le cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

conv_id = st.session_state["active_conv_id"]
session_id = st.session_state["session_id"]

# Display existing messages
msgs = db.get_messages(conv_id)
for m in msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        # Show sources for assistant messages (if available)
        if m["role"] == "assistant" and m.get("sources"):
            _display_sources(m["sources"])

# Chat input
prompt = st.chat_input("Pose ta question (réponse uniquement sur documents)…")

if prompt:
    request_id = generate_request_id()
    
    # Rate limiting
    allowed, retry_after = check_rate_limit(
        st.session_state,
        key="rate_limit_chat",
        max_requests=settings.rate_limit_max_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    if not allowed:
        st.warning(f"Trop de requêtes. Réessaie dans ~{retry_after}s.")
        log_error(request_id, session_id, "RATE_LIMITED")
        st.stop()

    # Save and display user message
    db.add_message(conv_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response (BUFFERED - no streaming to user)
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents + génération…"):
            with RequestTimer() as timer:
                try:
                    # Use buffered version - response is validated before we see it
                    answer, sources_meta, doc_ids, chunk_ids = answer_question_buffered(
                        prompt
                    )
                    is_refusal = answer == refusal()
                    
                except Exception as e:
                    logger.exception(
                        "Query failed",
                        extra={"request_id": request_id, "error_code": "QUERY_ERROR"},
                    )
                    log_error(request_id, session_id, "QUERY_ERROR")
                    answer = "Une erreur s'est produite. Veuillez réessayer."
                    sources_meta, doc_ids, chunk_ids = [], [], []
                    is_refusal = True

            # Audit log (allowlist only - no content!)
            log_query(
                request_id=request_id,
                session_id=session_id,
                doc_ids=doc_ids,
                chunk_ids=chunk_ids,
                scores=[s["score"] for s in sources_meta],
                latency_ms=timer.elapsed_ms,
                model=settings.openai_chat_model,
                verdict="refusal" if is_refusal else "answer",
            )

        # Display the validated response
        st.markdown(answer)

        # Display sources
        _display_sources(sources_meta)

    # Save assistant response with sources
    db.add_message(conv_id, "assistant", answer, sources=sources_meta)
    st.rerun()

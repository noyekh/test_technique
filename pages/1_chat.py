"""
Chatbot page - RAG-based Q&A interface.

SECURITY (v1.5):
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

st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")
username = require_auth()  # Bloque si non authentifié
st.title("💬 Chatbot interne (RAG strict)")

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
    labels = {c["conv_id"]: f"{c['title']} · {c['conv_id'][:6]}" for c in convs}

    def _fmt(cid: str) -> str:
        return labels.get(cid, f"Conversation {cid[:6]}")

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
        if st.button("🧹 Rafraîchir", use_container_width=True):
            st.rerun()

    st.divider()

    render_logout()
    st.caption(f"Connecté: {username}")

    # Security notice
    st.info(
        "🔒 **Mode sécurisé (v1.5)**\n\n"
        "Les réponses sont validées avant affichage. "
        "Le streaming est désactivé pour garantir la conformité."
    )

    st.caption(
        f"Rate limit: {settings.rate_limit_max_requests}/{settings.rate_limit_window_seconds}s"
    )

conv_id = st.session_state["active_conv_id"]
session_id = st.session_state["session_id"]

# Display existing messages
msgs = db.get_messages(conv_id)
for m in msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

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
        with st.expander("🔎 Sources"):
            if not sources_meta:
                st.write("Aucune source (refus ou score insuffisant).")
            else:
                for s in sources_meta:
                    st.write(
                        f"- **Source {s['i']}** — {s['source']} "
                        f"(chunk {s['chunk']}) — score {s['score']:.3f}"
                    )

    # Save assistant response
    db.add_message(conv_id, "assistant", answer)
    st.rerun()

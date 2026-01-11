"""
Chatbot page - RAG-based Q&A interface.

SECURITY:
- Streaming is DISABLED by default for legal contexts
- All responses are buffered and validated BEFORE display
- Audit logging with allowlist policy
"""

import logging
import os
import re
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
from backend.auth import render_logout, require_auth
from backend.logging_config import setup_logging
from backend.rag import (
    answer_question_buffered,
    refusal,
)
from backend.rag_runtime import contextualize_query
from backend.rate_limit import check_rate_limit
from backend.settings import settings

load_dotenv()
setup_logging()
db.init_db()

logger = logging.getLogger(__name__)


def _extract_cited_sources(answer: str) -> set[int]:
    """Extract source numbers cited in the answer [Source N]."""
    return {int(m.group(1)) for m in re.finditer(r"\[Source\s+(\d+)\]", answer, re.IGNORECASE)}


def _display_sources(sources: list[dict] | None, answer: str = "") -> None:
    """Display sources in a user-friendly expander. Grays out uncited sources."""
    cited = _extract_cited_sources(answer) if answer else set()
    with st.expander("🔎 Sources consultées"):
        if not sources:
            st.caption("Aucune source (refus ou documents insuffisants).")
        else:
            for s in sources:
                if not cited or s["i"] in cited:
                    st.markdown(f"📄 **[Source {s['i']}]** — *{s['source']}*")
                else:
                    st.caption(f"📄 [Source {s['i']}] — {s['source']} *(non citée)*")


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

    selected = st.selectbox("Choisir", options, index=options.index(current), format_func=_fmt)
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
                st.session_state["active_conv_id"] = (
                    remaining[0]["conv_id"] if remaining else db.ensure_default_conversation()
                )
                st.rerun()

    st.divider()

    # v1.11: Debug panel (contextualisation)
    ctx_debug = st.session_state.get("_last_ctx_debug")
    if ctx_debug:
        with st.expander("🐛 Debug"):
            if ctx_debug[0] == "reformulated":
                st.success(f"🔄 {ctx_debug[1]}")
            elif ctx_debug[0] == "unchanged":
                st.info(f"ℹ️ Pas de reformulation ({ctx_debug[1]} msgs)")
            elif ctx_debug[0] == "no_history":
                st.caption("ℹ️ Première question (pas d'historique)")
            elif ctx_debug[0] == "error":
                st.error(f"⚠️ {ctx_debug[1]}")
            if len(ctx_debug) >= 4:
                st.caption(f"📊 Sources: {ctx_debug[2]} | Docs: {ctx_debug[3]}")

    render_logout()
    st.caption(f"Connecté: {username}")

    # Technical details in expander (for developers)
    with st.expander("⚙️ Paramètres techniques"):
        st.caption(
            f"Rate limit: {settings.rate_limit_max_requests} req / {settings.rate_limit_window_seconds}s"
        )
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
for idx, m in enumerate(msgs):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        # Show sources for assistant messages (if available and not a refusal)
        if m["role"] == "assistant" and m.get("sources") and m["content"] != refusal():
            _display_sources(m["sources"], m["content"])
        # Show regenerate button for the last assistant message only
        if m["role"] == "assistant" and idx == len(msgs) - 1:
            if st.button(
                "🔄 Régénérer",
                key="regen_btn",
                help="Supprimer cette réponse et en générer une nouvelle",
            ):
                db.delete_last_assistant_message(conv_id)
                last_q = db.get_last_user_message(conv_id)
                if last_q:
                    st.session_state["_regen_prompt"] = last_q
                st.rerun()

# Chat input
prompt = st.chat_input("Pose ta question (réponse uniquement sur documents)…")

# Handle regeneration (prompt stored from button click)
regen_prompt = st.session_state.pop("_regen_prompt", None)
if regen_prompt:
    prompt = regen_prompt
    # Don't re-add user message since it already exists

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

    # Save and display user message (skip if regenerating)
    if not regen_prompt:
        db.add_message(conv_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate response (BUFFERED - no streaming to user)
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents + génération…"):
            ctx_debug = None  # Store debug info for display after spinner
            with RequestTimer() as timer:
                try:
                    # Build conversation history for context
                    history = [{"role": m["role"], "content": m["content"]} for m in msgs]

                    # v1.11: Contextualize query for retrieval
                    search_query = prompt
                    if history:
                        try:
                            search_query = contextualize_query(prompt, history)
                            if search_query != prompt:
                                ctx_debug = ("reformulated", search_query)
                            else:
                                ctx_debug = ("unchanged", len(history))
                        except Exception as ctx_err:
                            ctx_debug = ("error", str(ctx_err))

                    # Use buffered version:
                    # - search_query for retrieval (contextualized)
                    # - history for LLM context (original conversation)
                    answer, sources_meta, doc_ids, chunk_ids = answer_question_buffered(
                        search_query,
                        history=history,
                    )
                    is_refusal = answer == refusal()

                except Exception:
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

        # Store debug in session for persistence (v1.11)
        # Always store retrieval stats, even without contextualization
        if ctx_debug:
            debug_info = (*ctx_debug, len(sources_meta), len(doc_ids))
        else:
            debug_info = ("no_history", 0, len(sources_meta), len(doc_ids))
        st.session_state["_last_ctx_debug"] = debug_info

        # Display the validated response
        st.markdown(answer)

        # Display sources (only if not a refusal)
        if not is_refusal:
            _display_sources(sources_meta, answer)

    # Save assistant response with sources
    db.add_message(conv_id, "assistant", answer, sources=sources_meta)
    st.rerun()

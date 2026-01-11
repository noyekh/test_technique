"""
Documents page - Upload, manage, and vectorize documents.

SECURITY:
- Audit logging for all operations
- Verified end-to-end deletion
- Session ID tracking
"""

import logging
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from backend import db
from backend.audit_log import generate_request_id, log_upload
from backend.auth import render_logout, require_auth
from backend.documents import delete_document_complete
from backend.files import save_upload
from backend.ingest import read_to_text
from backend.logging_config import setup_logging
from backend.mime_validation import validate_mime
from backend.rag import add_doc_chunks, chunk_text
from backend.settings import settings

load_dotenv()
setup_logging()
db.init_db()

logger = logging.getLogger(__name__)

# Note: st.set_page_config() est dans main.py (st.navigation)
username = require_auth()  # Bloque si non authentifié
st.title("📄 Gestion des documents")

with st.sidebar:
    render_logout()
    st.caption(f"Connecté: {username}")

# Check required API keys (show error only if missing)
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY manquante — l'indexation échouera.")
if not os.getenv("VOYAGE_API_KEY"):
    st.error("❌ VOYAGE_API_KEY manquante — l'indexation échouera.")

# Technical details in expander
with st.expander("⚙️ Paramètres techniques", expanded=False):
    openai_key = os.getenv("OPENAI_API_KEY")
    voyage_key = os.getenv("VOYAGE_API_KEY")
    st.caption(f"OpenAI: {'✅ configurée' if openai_key else '❌ manquante'}")
    st.caption(f"Voyage: {'✅ configurée' if voyage_key else '❌ manquante'}")
    st.caption(f"Taille max: {settings.max_file_size_mb} MB")
    if st.button("🔄 Vider le cache", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# Initialize session ID for audit
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex[:16]

session_id = st.session_state["session_id"]

# Track last successful upload to prevent re-processing after rerun
if "last_uploaded_sha256" not in st.session_state:
    st.session_state["last_uploaded_sha256"] = None

# Show success message if we just uploaded
if "upload_success_msg" in st.session_state:
    st.success(st.session_state.pop("upload_success_msg"))

MAX_FILE_SIZE = settings.max_file_size_mb * 1024 * 1024

# File upload section - key changes after successful upload to clear widget
upload_key = st.session_state.get("upload_key", "uploader_0")
uploaded_files = st.file_uploader(
    "Uploader (.txt, .csv, .html)",
    type=["txt", "csv", "html", "htm"],
    key=upload_key,
    accept_multiple_files=True,
)

# Process all uploaded files
success_count = 0
success_messages = []

for uploaded in uploaded_files:
    request_id = generate_request_id()

    try:
        raw = uploaded.getvalue()

        # Check file size
        if len(raw) > MAX_FILE_SIZE:
            mb = len(raw) // 1024 // 1024
            st.error(f"❌ {uploaded.name}: Trop volumineux ({mb} MB > {settings.max_file_size_mb} MB)")
            continue

        # Save file to disk
        digest, stored_path, ext, size_bytes = save_upload(uploaded.name, raw)

        # Validate MIME type
        ok_mime, detected = validate_mime(ext, raw)
        if not ok_mime:
            try:
                stored_path.unlink(missing_ok=True)
            except OSError:
                pass
            st.error(f"❌ {uploaded.name}: Type MIME invalide. Détecté: {detected}")
            continue

        # Check for duplicates
        existing = db.get_document_by_sha256(digest)
        if existing:
            try:
                stored_path.unlink(missing_ok=True)
            except OSError:
                pass
            st.warning(f"⚠️ {uploaded.name}: Déjà indexé (même contenu). Ignoré.")
            continue

        # Parse and chunk
        text = read_to_text(ext, raw)
        chunks = chunk_text(text)

        # Generate doc_id before DB insertion for vectorization
        doc_id = uuid.uuid4().hex

        # Vectorize FIRST (before DB insert) - fail fast
        with st.status(f"Vectorisation de {uploaded.name}…", expanded=False):
            try:
                chunk_ids = add_doc_chunks(doc_id, uploaded.name, chunks)
            except Exception as vec_error:
                # Cleanup file on vectorization failure
                try:
                    stored_path.unlink(missing_ok=True)
                except OSError:
                    pass
                st.error(f"❌ {uploaded.name}: Erreur vectorisation - {vec_error}")
                continue

        # Add to database AFTER successful vectorization
        db.add_document_with_id(
            doc_id=doc_id,
            original_name=uploaded.name,
            stored_path=str(stored_path),
            ext=ext,
            sha256=digest,
            size_bytes=size_bytes,
        )
        db.add_chunks(doc_id, chunk_ids)

        # Audit log (no content!)
        log_upload(
            request_id=request_id,
            session_id=session_id,
            doc_id=doc_id,
            chunk_ids=chunk_ids,
        )

        logger.info(
            "Uploaded and indexed document",
            extra={"doc_id": doc_id, "chunks": len(chunk_ids)},
        )

        success_count += 1
        success_messages.append(f"'{uploaded.name}' ({len(chunk_ids)} chunks)")

    except Exception as e:
        logger.exception("Ingestion failed", extra={"error_code": "INGESTION_ERROR"})
        st.error(f"❌ {uploaded.name}: {e}")

# After processing all files, show summary and clear uploader
if success_count > 0:
    st.session_state["upload_success_msg"] = f"✅ {success_count} document(s) indexé(s): {', '.join(success_messages)}"
    st.session_state["upload_key"] = f"uploader_{uuid.uuid4().hex[:8]}"
    st.rerun()

st.divider()
st.subheader("Documents existants")

# List existing documents
docs = db.list_documents()
if not docs:
    st.info("Aucun document pour le moment.")
else:
    for d in docs:
        col1, col2 = st.columns([0.78, 0.22])
        with col1:
            st.write(f"**{d['original_name']}**")
            with st.expander("Détails techniques"):
                st.code(
                    f"ID: {d['doc_id']}\n"
                    f"SHA256: {d['sha256']}\n"
                    f"Taille: {d['size_bytes']} bytes\n"
                    f"Ext: .{d['ext']}\n"
                    f"Créé: {d['created_at']}"
                )
        with col2:
            if st.button(
                "🗑️ Supprimer", key=f"del_{d['doc_id']}", use_container_width=True
            ):
                try:
                    result = delete_document_complete(d["doc_id"], session_id)

                    if result.success:
                        st.success("Supprimé (fichier + DB + index) ✅")
                    elif result.partial:
                        st.warning(
                            f"Suppression partielle. Erreurs: {', '.join(result.errors)}"
                        )
                    else:
                        st.error(
                            f"Échec suppression. Erreurs: {', '.join(result.errors)}"
                        )

                    st.rerun()
                except Exception as e:
                    logger.exception(
                        "Deletion failed",
                        extra={"doc_id": d["doc_id"], "error_code": "DELETE_ERROR"},
                    )
                    st.error(f"Erreur suppression: {e}")

# Show deletion history
st.divider()
with st.expander("🗄️ Historique des suppressions (audit RGPD)"):
    deletions = db.list_deletions(limit=20)
    if not deletions:
        st.info("Aucune suppression enregistrée.")
    else:
        for d in deletions:
            st.write(
                f"- **{d['doc_id'][:8]}...** — {d['chunk_count']} chunks — {d['deleted_at']}"
            )

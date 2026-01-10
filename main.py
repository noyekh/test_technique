import streamlit as st
from dotenv import load_dotenv

from backend.auth import require_auth, render_logout
from backend.db import init_db
from backend.logging_config import setup_logging

load_dotenv()
setup_logging()
init_db()

st.set_page_config(page_title="Legal RAG PoC", page_icon="⚖️", layout="wide")
username = require_auth()

st.title("⚖️ Legal RAG PoC — Cabinet Emilia Parenti")

with st.sidebar:
    render_logout()
    st.caption(f"Connecté: {username}")

st.markdown(
    """
PoC interne : chatbot **strictement basé sur les documents uploadés**.

- Va dans **Documents** pour uploader / supprimer et vectoriser
- Puis dans **Chatbot** pour poser des questions

### Architecture v1.8

- ✅ **Authentification** : login/password avec audit
- ✅ **Pas de streaming** : réponses validées avant affichage
- ✅ **Logs minimalistes** : politique allowlist stricte
- ✅ **Suppression vérifiable** : audit trail RGPD
- ✅ **Prompt durci** : documents traités comme données non fiables
"""
)

st.info("Conseil : uploade 2–3 documents, puis teste une question précise.")

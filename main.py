"""
Point d'entrée Streamlit avec navigation personnalisée.

Le cahier des charges demande 2 pages :
- Page 1 : Chatbot
- Page 2 : Documents

st.navigation() permet de contrôler exactement les pages affichées
dans la sidebar (Streamlit >= 1.36).
"""

import streamlit as st

st.set_page_config(page_title="Legal RAG PoC", page_icon="⚖️", layout="wide")

# Navigation personnalisée : seulement 2 pages visibles
pages = st.navigation(
    [
        st.Page("pages/1_chat.py", title="Chatbot", icon="💬", default=True),
        st.Page("pages/2_documents.py", title="Documents", icon="📄"),
    ]
)
pages.run()

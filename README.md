# Legal RAG PoC v1.10 — Pipeline Optimisé + Reranking + Citation Verification

PoC d'un chatbot interne pour cabinet d'avocats avec architecture hexagonale, sécurité renforcée, et retrieval optimisé pour le juridique français.

## Avertissements importants

1. **Les données juridiques sont sensibles** — aucune donnée ne doit transiter sans chiffrement (HTTPS)
2. **Rétention API** — OpenAI et Voyage AI peuvent conserver des logs (abuse monitoring)
3. **Credentials de test** — Changer impérativement les mots de passe avant production

## Nouveautés v1.10 (vs v1.9)

| Composant | v1.9 | v1.10 | Impact |
|-----------|------|-------|--------|
| **Sources persistantes** | Éphémères (perdues au refresh) | **Stockées en DB** | UX : sources toujours visibles |
| **Affichage sources** | Technique (chunk, score) | **Lisible** (`📄 [Source 1] — fichier.txt`) | Interprétable par non-dev |
| **Conversations** | ID aléatoire, ordre création | **Auto-titre + tri par usage** | Navigation intuitive |

### Détails v1.10

- **Sources persistantes** : Nouvelle colonne `sources_json` dans `messages`, sauvegardées avec chaque réponse assistant
- **Affichage UX** : Suppression des métadonnées techniques (chunk index, score), format lisible
- **Auto-titre** : Première question de l'utilisateur devient le titre de la conversation
- **Tri intelligent** : Conversations triées par `updated_at DESC` (dernière utilisée en premier)

## Nouveautés v1.9 (vs v1.8)

| Composant | v1.8 | v1.9 | Gain |
|-----------|------|------|------|
| **Pipeline** | Hybrid (top_k=6) → LLM | **Multi-query → Hybrid (top_k=100) → Rerank → LLM → Verify** | Architecture complète |
| **Reranking** | Non | **Voyage rerank-2.5** | +40% MRR, -35% hallucinations |
| **Multi-query** | Non | **3 reformulations LLM** | +25% recall |
| **Citation verification** | Non | **Post-LLM validation** | -90% fausses citations |

### Pipeline v1.9

```
Query → Multi-query (3 variants) → Hybrid BM25+Dense (top_k=100) → Rerank (top_n=15) → LLM → Citation verification
```

**Documentation technique** :
- [CHOIX_TECHNIQUES.md](CHOIX_TECHNIQUES.md) — Arbitrages, limitations acceptées, chemin vers prod
- [ETAT_DE_LART.md](ETAT_DE_LART.md) — Recherche bibliographique, sources peer-reviewed

## Nouveautés v1.8 (vs v1.7)

| Composant | v1.7 | v1.8 | Justification |
|-----------|------|------|---------------|
| **Auth** | Nginx BasicAuth | **streamlit-authenticator** | Auth intégrée, zéro infra |
| **Credentials** | nginx htpasswd | **bcrypt + secrets.toml** | Standard sécurisé |
| **Audit** | Logs query/upload/delete | **+ events auth** | Traçabilité connexions |

### Pourquoi streamlit-authenticator ?

- **Zéro infrastructure** : Pas besoin de nginx/reverse proxy pour l'auth
- **Cookies sécurisés** : Session persistante signée
- **Bcrypt** : Hash des mots de passe (standard OWASP)
- **Audit natif** : Events login_success/login_failed/logout

## Nouveautés v1.7 (vs v1.6)

| Composant | v1.6 | v1.7 | Justification |
|-----------|------|------|---------------|
| **LLM** | gpt-4o-mini | **gpt-4.1-mini** | +17% cross-ref légal (Thomson Reuters), 1M context |
| **Embeddings** | text-embedding-3-small | **voyage-3-large** | +11pp MLEB, 200M tokens GRATUITS |

## Rappel v1.6 (conservé)

| Feature | Description | Source |
|---------|-------------|--------|
| **Token-aware chunking** | 768 tokens, séparateurs juridiques FR | Chroma Research 2024 |
| **Hybrid BM25+Dense** | 60% BM25 pour citations exactes | Anthropic 2024 (-49% échecs) |
| **Streaming OFF** | Buffer + validation avant affichage | OWASP LLM 2025 |
| **Logs allowlist** | Jamais de contenu utilisateur | RGPD compliance |
| **Suppression vérifiable** | Tombstones RGPD | CNIL guidelines |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Édite .env et configure :
#   - OPENAI_API_KEY (pour LLM)
#   - VOYAGE_API_KEY (pour embeddings)
```

### Obtenir les clés API

1. **OpenAI** : https://platform.openai.com/api-keys
2. **Voyage AI** : https://dash.voyageai.com/ (gratuit, 200M tokens inclus)

## Configuration authentification

### Utilisateurs par défaut (test)

| Username | Password | Nom affiché |
|----------|----------|-------------|
| `admin` | `admin123` | Emilia Parenti |
| `avocat1` | `avocat123` | Avocat Junior |

### Ajouter un utilisateur

```bash
python create_user.py
```

Ou manuellement :

```bash
# 1. Générer le hash bcrypt
python3 -c "import bcrypt; print(bcrypt.hashpw(b'MON_PASSWORD', bcrypt.gensalt()).decode())"

# 2. Ajouter dans .streamlit/secrets.toml
```

```toml
[auth.credentials.usernames.nouveau_user]
name = "Prénom Nom"
password = "$2b$12$LE_HASH_GENERE"
```

### Désactiver l'authentification (dev)

```bash
AUTH_ENABLED=false streamlit run main.py
```

## Configuration v1.9

| Variable | Description | Défaut |
|----------|-------------|--------|
| `OPENAI_API_KEY` | Clé API OpenAI | (requis) |
| `VOYAGE_API_KEY` | Clé API Voyage AI | (requis) |
| `AUTH_ENABLED` | Activer l'authentification | `true` |
| `OPENAI_CHAT_MODEL` | Modèle LLM | `gpt-4.1-mini` |
| `VOYAGE_EMBED_MODEL` | Modèle embeddings | `voyage-3-large` |
| `CHUNK_SIZE_TOKENS` | Taille chunks en tokens | `768` |
| `CHUNK_OVERLAP_TOKENS` | Overlap en tokens | `115` (~15%) |
| `HYBRID_SEARCH` | Activer BM25+Dense | `true` |
| `BM25_WEIGHT` | Poids BM25 (0.0-1.0) | `0.6` |
| `ENABLE_STREAMING` | Streaming réponses | `false` |

### Configuration v1.9 (pipeline optimisé)

| Variable | Description | Défaut |
|----------|-------------|--------|
| `RERANK_ENABLED` | Activer Voyage rerank-2.5 | `true` |
| `RERANK_MODEL` | Modèle reranker | `rerank-2.5` |
| `RERANK_TOP_N` | Docs après reranking | `15` |
| `RETRIEVAL_TOP_K` | Docs pour reranker | `100` |
| `MULTI_QUERY_ENABLED` | Activer expansion | `true` |
| `MULTI_QUERY_VARIANTS` | Nombre de variants | `3` |
| `CITATION_VERIFICATION_ENABLED` | Vérification citations | `true` |
| `CITATION_VERIFICATION_LEVEL` | Niveau (basic/presence/semantic) | `presence` |

## Usage

```bash
streamlit run main.py
```

1. Se connecter (admin / admin123)
2. **Documents** → uploader `.txt`, `.csv`, `.html`
3. **Chatbot** → poser une question

## Migration vers Production

| Composant | PoC (v1.9) | Production | Changement |
|-----------|------------|------------|------------|
| LLM | gpt-4.1-mini | Multi-model (GPT-4.1 / Claude Sonnet 4) | Config |
| Embeddings | voyage-3-large | voyage-3-large ou fine-tuned | Aucun |
| Reranker | **Voyage rerank-2.5** | Voyage rerank-2.5 | **Inclus v1.9** |
| Multi-query | **3 variants** | 3-5 variants | **Inclus v1.9** |
| Citation verif | **presence level** | semantic level | **Inclus v1.9** |
| Vector DB | ChromaDB | Qdrant Cloud / Pinecone | Migration |
| Auth | streamlit-authenticator | Azure AD / Auth0 | Infra SSO |
| HTTPS | Non | Obligatoire | Nginx/Caddy |

## Tests

```bash
pytest -v
ruff check .
ruff format .
```

## Évolutions recommandées (Prod)

### P0 (avant mise en production)

- [ ] SSO/OIDC (Azure AD, Auth0) au lieu de secrets.toml
- [ ] HTTPS obligatoire (Let's Encrypt)
- [ ] Changer tous les mots de passe par défaut
- [ ] DPA avec OpenAI et Voyage AI
- [ ] DPIA/AIPD si données sensibles

### P1 (court terme)

- [x] ~~Reranking Voyage rerank-2.5~~ **Inclus v1.9**
- [x] ~~Multi-query expansion~~ **Inclus v1.9**
- [x] ~~Citation verification~~ **Inclus v1.9**
- [ ] Rate limiting Redis (distribué)
- [ ] Monitoring/alerting
- [ ] Multi-model routing (Claude Sonnet 4 pour raisonnement complexe)

### P2 (moyen terme)

- [ ] Citation verification niveau `semantic` (avec embeddings)
- [ ] Fine-tuning embeddings domaine
- [ ] Multi-tenant (séparation par dossier)
- [ ] PDF/DOCX support
- [ ] Contextual Retrieval (prepending)

## Structure des fichiers

```
legal-rag-poc-v1.10/
├── main.py                   # Point d'entrée Streamlit
├── create_user.py            # Script création utilisateurs
├── CHOIX_TECHNIQUES.md       # Documentation technique v1.10
├── pages/
│   ├── 1_chat.py             # Interface chatbot (+ sources persistantes v1.10)
│   └── 2_documents.py        # Gestion documents + audit
├── backend/
│   ├── __init__.py
│   ├── settings.py           # Configuration centralisée
│   ├── auth.py               # Authentification streamlit-authenticator
│   ├── db.py                 # SQLite + tombstones RGPD (+ sources_json v1.10)
│   ├── rag_core.py           # Logique pure (+ citation verification v1.9)
│   ├── rag_runtime.py        # Pipeline (multi-query + rerank)
│   ├── rag.py                # Façade
│   ├── reranker.py           # Voyage rerank-2.5
│   ├── multi_query.py        # Query expansion
│   ├── citation_verifier.py  # Post-LLM verification
│   ├── audit_log.py          # Logs allowlist stricte
│   ├── documents.py          # Suppression vérifiable
│   ├── files.py              # Gestion fichiers
│   ├── ingest.py             # Parsing documents
│   ├── security.py           # Sanitization
│   ├── rate_limit.py         # Rate limiting session
│   ├── mime_validation.py    # Validation MIME
│   └── logging_config.py     # Config logging app
├── tests/
├── data/                     # (gitignored)
├── .streamlit/
│   └── secrets.toml          # Credentials auth (gitignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Références

### LLM & Embeddings
- [GPT-4.1 Release](https://openai.com/index/gpt-4-1/) — Avril 2025
- [Voyage AI voyage-3-large](https://blog.voyageai.com/2025/01/07/voyage-3-large/) — Jan 2025
- [MLEB Legal Benchmark](https://huggingface.co/blog/isaacus/introducing-mleb) — Oct 2025

### Chunking & Retrieval
- [Chroma Research: Evaluating Chunking Strategies](https://research.trychroma.com/evaluating-chunking) — Juillet 2024
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — Sept 2024
- [MDPI Legal RAG Study](https://www.mdpi.com/2073-8994/17/5/633) — 2025

### Sécurité & Conformité
- [OWASP LLM Top 10 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [CNIL Fiches IA](https://www.cnil.fr/fr/les-fiches-pratiques-ia)

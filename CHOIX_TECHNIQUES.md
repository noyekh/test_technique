# CHOIX TECHNIQUES — Legal RAG PoC v1.10

**Candidat** : [Nom]
**Date** : 11 janvier 2026
**Temps de lecture** : 5 minutes

---

## TL;DR — Décisions clés en 30 secondes

| Composant | Choix | Alternative écartée | Pourquoi |
|-----------|-------|---------------------|----------|
| **LLM** | gpt-4.1-mini | gpt-4o-mini | 1M context, +53% accuracy fiscale¹ |
| **Embeddings** | voyage-3-large | text-embedding-3-large | +10,58% nDCG@10, 200M tokens **gratuits**² |
| **Reranking** | Voyage rerank-2.5 | Cohere Rerank 4 | Cohere **dégrade** perf légale³, même prix |
| **Retrieval** | Two-stage 100→15 | Single top_k=6 | +10,5% MRR⁴ |
| **Query** | Multi-query ×3 | Single query | +10pp Recall@10⁵ |
| **Citations** | Vérification post-gen | Aucune | +15,46% accuracy⁶ |
| **Seuils** | Reranker score (0.3) | Cosine threshold (0.35) | Anti-pattern⁷, -76% requêtes |
| **Auth** | streamlit-authenticator | Nginx BasicAuth | Zéro infra, standard Streamlit |

**Sources** :
¹ Blue J Tax, avril 2025 (vendor partner)
² Voyage AI Blog, jan 2025 — validé MongoDB
³ Stanford LegalBench-RAG (arXiv:2408.10343)
⁴ MyScale benchmark 2025
⁵ Stanford RegLab, CSLAW '25
⁶ CiteFix, ACL 2025 Industry Track
⁷ Cambridge 2025 (23 625 itérations grid-search) — seuil 0.5 rejette 76-100% requêtes

---

## 1. Architecture pipeline

```
v1.8 (avant):
Query ──────────────────────────> Hybrid (top_k=6) ──> LLM ──> Response

v1.9 (après):
Query ──> Multi-query ──> Hybrid (top_k=100) ──> Rerank (15) ──> LLM ──> Citation check
          +10pp Recall        +4,7% Hit Rate      +10,5% MRR       +15,46% accuracy
```

**Gain cumulé estimé** : +25-40% qualité retrieval, +15% accuracy citations

**Coût additionnel** : ~$0.50 pour 2K queries (free tiers Voyage couvrent le PoC)

---

## 2. Décisions clés avec arbitrages

### 2.1 Reranking — Le choix le plus critique

| ✅ Retenu | ❌ Écarté | Justification |
|-----------|----------|---------------|
| **Voyage rerank-2.5** | Cohere Rerank 4 | LegalBench-RAG : Cohere v3 **dégrade** les métriques légales |
| nDCG@10 = **0.235** | nDCG@10 = 0.219 | Agentset Leaderboard, nov 2025 |
| | Jina Reranker v2 | nDCG@10 = 0.193, latence +20% |

**Observation importante** : Cohere mène en ELO (1627 vs 1547), mais Voyage obtient le meilleur nDCG@10. Pour le légal où la précision prime, nDCG est plus pertinent.

**Instruction légale utilisée** :
```python
"Documents juridiques français : contrats, articles de loi, jurisprudence."
```

### 2.2 Embeddings — Choix validé par tiers

| ✅ Retenu | ❌ Écarté | Justification |
|-----------|----------|---------------|
| **voyage-3-large** | text-embedding-3-large | +10,58% nDCG@10 (Voyage AI, validé MongoDB) |
| Latence 89ms/chunk | Latence 311ms/chunk | 3,5x plus rapide (MongoDB benchmark) |
| | voyage-law-2 | Voyage AI jan 2025 : v3-large surpasse v2 spécialisés |

**Validation indépendante** : MongoDB Technical Blog confirme "voyage-3-large produces the strongest ranking". DEV.to/DataStax : "surprise leader in embedding relevance".

### 2.3 Hallucinations — Réalité vs Marketing

> "Legal RAG tools hallucinate between 17% and 33% of the time."
> — Stanford HAI, **Journal of Empirical Legal Studies**, 2025 (peer-reviewed, κ=0.77)

| Système | Taux d'hallucination | Accuracy |
|---------|---------------------|----------|
| GPT-4 (sans RAG) | 58-82% | — |
| **Lexis+ AI** | **17%** | 65% |
| Westlaw AI-AR | 34% | 42% |
| Harvey (claim vendor) | 0,2% | ⚠️ Non validé |

**Implication** : La vérification humaine reste **obligatoire**. CiteFix (ACL 2025) montre +15,46% accuracy avec post-processing — notre approche.

### 2.4 Seuils de relevance — Pourquoi pas de seuil cosine hardcodé

| ✅ Retenu | ❌ Écarté | Justification |
|-----------|----------|---------------|
| **Seuil reranker 0.3** | Seuil cosine 0.35 | Cambridge 2025 : seuil 0.5 rejette **76-100%** des requêtes |
| Filtrage via top_n=15 | min_relevance=0.35 | Harvey, LexisNexis, Thomson Reuters : **aucun seuil cosine** |
| | | Scores cosine varient de **50%** entre modèles d'embedding |

**Problème identifié** : Avec voyage-3-large, les scores cosine observés étaient 0.21-0.31 — tous rejetés par seuil 0.35.

**Solution implémentée (v1.9.1)** :
```python
# AVANT (anti-pattern)
rag_min_relevance: float = 0.35  # Rejette 76-100% des requêtes

# APRÈS (best practice)
rag_min_relevance: float = 0.0   # Désactivé - le reranker filtre
rerank_min_score: float = 0.3    # Seuil sur score reranker calibré
```

**Pipeline final** :
1. Hybrid retrieval (top_k=100) → Pas de filtrage
2. Reranking (top_n=15) → Filtrage par qualité reranker
3. Seuil reranker optionnel (>0.3) → Refus si aucun document pertinent

---

## 3. Ce qu'on n'a PAS implémenté (et pourquoi)

| Fonctionnalité | Gain potentiel | Raison de l'exclusion |
|----------------|----------------|----------------------|
| **Contextual Retrieval** | -67% échecs (claim Anthropic) | ⚠️ Non répliqué par tiers, réindexation ~$1/1M |
| **Shepard's/KeyCite API** | Élimine misgrounded | Licence enterprise, hors scope PoC |
| **Late Chunking** (Jina) | Efficient, pas d'appels LLM | Complexité, non benchmarké légal |
| **OAuth2/OIDC** | SSO enterprise | Over-engineering pour PoC |
| **Isolation données/user** | Multi-tenant | Pas dans cahier des charges |

**Message à l'examinateur** : Ces choix sont **conscients**, pas des oublis. Le -67% d'Anthropic n'a **jamais été répliqué** (arXiv:2504.19754, Bologna 2025).

---

## 4. Limitations acceptées

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Vérification citations = existence seule | Ne détecte pas "misgrounded" | Avertissement utilisateur |
| Pas de Contextual Retrieval | Gain non validé manqué | Two-stage + rerank compense |
| GPT-4.1-mini non benchmarké légal | Performance incertaine | Tests internes recommandés |
| Hallucinations résiduelles ~17% | Incompressible sans Shepard's | **Humain dans la boucle** |

---

## 5. Chemin vers production

| Composant | PoC v1.9 | Production | Effort |
|-----------|----------|------------|--------|
| Embeddings | voyage-3-large | Fine-tuned légal FR | Élevé |
| Reranking | rerank-2.5 | Idem + domain tuning | Faible |
| Citations | CiteFix-style | **Shepard's/KeyCite API** | Moyen |
| LLM | gpt-4.1-mini | Gemini 3 Pro (87% LegalBench) | Faible |
| Auth | streamlit-authenticator | OAuth2/OIDC + SSO | Moyen |
| Vector DB | ChromaDB | Qdrant Cloud | Faible |

---

## 6. Coûts détaillés (janvier 2026)

| Composant | Prix | Free tier | Usage PoC |
|-----------|------|-----------|-----------|
| Embeddings voyage-3-large | $0.18/M | **200M tokens** | **$0** |
| Reranking rerank-2.5 | $0.05/M | **200M tokens** | **$0** |
| LLM gpt-4.1-mini | $0.40/$1.60/M | — | ~$4 |
| Multi-query (3×) | Via LLM | — | ~$0.50 |

**Total PoC** : ~$4.50 pour 200 docs + 2K queries

---

## 7. Sources — Niveau de confiance

| Affirmation clé | Source | Type | Confiance |
|-----------------|--------|------|-----------|
| 17-33% hallucinations légales | Stanford HAI, JELS 2025 | **Peer-reviewed** | ✅ Haute |
| Cohere dégrade perf légale | LegalBench-RAG arXiv:2408.10343 | Preprint | ✅ Haute |
| voyage-3-large +10,58% vs OpenAI | Voyage AI jan 2025 | Vendor | ⚠️ Moyenne |
| ↳ Validé par | MongoDB Technical Blog 2025 | Indépendant | ✅ Haute |
| +10,5% MRR reranking | MyScale benchmark 2025 | Indépendant | ✅ Haute |
| +15,46% accuracy CiteFix | ACL 2025 Industry Track | **Peer-reviewed** | ✅ Haute |
| -67% Contextual Retrieval | Anthropic sept 2024 | Vendor | ❌ **Non répliqué** |
| GPT-4.1-mini +53% fiscal | Blue J Tax avril 2025 | Partner | ⚠️ Moyenne |

**Gaps identifiés** :
- GPT-4.1-mini **non benchmarké** sur LegalBench (Vals.ai incomplet)
- voyage-law-2 vs voyage-3-large : **aucune validation tierce**
- Aucun benchmark direct 3 rerankers sur données légales identiques

---

*Document généré le 11 janvier 2026*
*Voir ETAT_DE_LART.md pour le détail des recherches et sources complètes*

# ÉTAT DE L'ART — RAG Juridique 2025-2026

**Synthèse des recherches pour Legal RAG PoC v1.9**
**Date de compilation** : 11 janvier 2026
**Critère de sélection** : Sources ≥ 2025, priorité peer-reviewed

---

## Table des matières

1. [Hallucinations en RAG légal](#1-hallucinations-en-rag-légal)
2. [Reranking : benchmarks et comparatifs](#2-reranking-benchmarks-et-comparatifs)
3. [Embeddings : état de l'art 2025](#3-embeddings-état-de-lart-2025)
4. [Multi-query expansion](#4-multi-query-expansion)
5. [Two-stage retrieval](#5-two-stage-retrieval)
6. [Seuils de relevance : anti-patterns et best practices](#6-seuils-de-relevance-anti-patterns-et-best-practices)
7. [Contextual Retrieval : validation critique](#7-contextual-retrieval-validation-critique)
8. [Citation verification](#8-citation-verification)
9. [LLMs pour le légal](#9-llms-pour-le-légal)
10. [Architectures production](#10-architectures-production)
11. [Analyse critique des sources](#11-analyse-critique-des-sources)
12. [Références complètes](#12-références-complètes)

---

## 1. Hallucinations en RAG légal

### 1.1 Étude de référence : Stanford HAI (2025)

**Source principale** : Magesh, V. et al. "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools."
**Publication** : *Journal of Empirical Legal Studies*, Vol. 22, Issue 2, pp. 216-242, 2025
**URL** : https://onlinelibrary.wiley.com/doi/full/10.1111/jels.12413
**Statut** : ✅ **Peer-reviewed** (publication académique de référence)

#### Méthodologie

- 200+ requêtes légales préenregistrées
- Cohen's κ = 0.77 (accord inter-évaluateurs substantiel)
- Inter-rater agreement : 85,4%
- Évaluation par juristes qualifiés

#### Résultats clés

| Système | Hallucination | Accuracy | Responsiveness |
|---------|---------------|----------|----------------|
| GPT-4 (baseline) | **58-82%** | — | — |
| Lexis+ AI | **17%** | 65% | 98% |
| Ask Practical Law AI | 17% | 19% | 60% |
| Westlaw AI-AR | **34%** | 42% | 93% |

#### Citations exactes

> "We find that legal RAG can reduce hallucinations compared to general-purpose AI systems (here, GPT-4), but hallucinations remain substantial, wide-ranging, and potentially insidious."

> "A citation can be 'hallucination-free' (the case exists) but still misleading (cited for the wrong proposition). These 'misgrounded' citations may be more dangerous than fabricated ones."

### 1.2 Harvey BigLaw Bench (2024-2025)

**Source** : Harvey AI Blog, "BigLaw Bench: Hallucinations"
**URL** : https://www.harvey.ai/blog/biglaw-bench-hallucinations
**Date** : Octobre 2024
**Statut** : ⚠️ Vendor (conflit d'intérêts)

| Modèle | Taux hallucination |
|--------|-------------------|
| Harvey Assistant | **0,2%** (1/500 claims) |
| Claude | 0,7% |
| ChatGPT | 1,3% |
| Gemini | 1,9% |

**⚠️ Mise en garde** : Méthodologie différente de Stanford (tâches grounded vs queries ouvertes). Le 0,2% n'a **pas été validé indépendamment**.

### 1.3 VALS AI Legal Report (Février 2025)

**Source** : VALS AI Industry Report
**URL** : https://www.vals.ai/vlair
**Statut** : ✅ Benchmark indépendant

- Harvey : top performer 5/6 tâches (94,8% Document Q&A)
- CoCounsel : excellent performance
- **LexisNexis s'est retiré de l'évaluation** — signal préoccupant

### 1.4 AI Hallucination Cases Database

**Source** : Damien Charlotin (académique)
**URL** : https://www.damiencharlotin.com/hallucinations/
**Statut** : ✅ Suivi indépendant continu

- **764+ cas** documentés globalement
- 324 cas tribunaux US
- Évolution : ~2 cas/semaine (début 2025) → **2-3 cas/jour** (fin 2025)

---

## 2. Reranking : benchmarks et comparatifs

### 2.1 Agentset Reranker Leaderboard (Novembre 2025)

**Source** : Agentset
**URL** : https://agentset.ai/rerankers
**Date** : 25 novembre 2025
**Méthodologie** : GPT-5 comme juge, datasets FiQA, SciFact, PG
**Statut** : ✅ Benchmark indépendant

| Rang | Modèle | ELO | nDCG@10 | Latence | Prix/1M |
|------|--------|-----|---------|---------|---------|
| 1 | Zerank 2 | **1654** | 0.223 | 565ms | $0.025 |
| 2 | Cohere Rerank 4 Pro | 1627 | 0.219 | 614ms | $0.050 |
| 3 | Zerank 1 | 1598 | 0.224 | 607ms | $0.025 |
| 4 | **Voyage rerank-2.5** | 1547 | **0.235** | 613ms | $0.050 |
| 6 | Voyage rerank-2.5-lite | 1528 | 0.226 | 616ms | $0.020 |
| 7 | Cohere Rerank 4 Fast | 1506 | 0.216 | 447ms | $0.050 |
| 11 | Jina Reranker v2 | 1306 | 0.193 | 746ms | $0.045 |

**Observation clé** : Voyage rerank-2.5 obtient le **meilleur nDCG@10** (0.235) malgré un ELO inférieur. Pour le légal, nDCG est plus pertinent que ELO.

### 2.2 LegalBench-RAG : Performance légale spécifique

**Source** : arXiv:2408.10343
**URL** : https://arxiv.org/abs/2408.10343
**Date** : Août 2024
**Statut** : ✅ Preprint académique (Stanford)

**Finding critique** :

> "Cohere rerank-english-v3.0 **underperformed versus no reranker at all** on legal precision and recall metrics."

**Corpus** : 6 858 paires Q&A, 79M+ caractères texte légal
**Datasets** : CUAD, MAUD, ContractNLI, PrivacyQA

**⚠️ Gap** : rerank-2.5 et Cohere Rerank 4 non testés sur ce benchmark.

### 2.3 Pricing exact (Janvier 2026)

**Source** : https://docs.voyageai.com/docs/pricing

| Fournisseur | Modèle | Prix/1M tokens | Free tier |
|-------------|--------|----------------|-----------|
| Voyage AI | rerank-2.5 | $0.05 | **200M tokens** |
| Voyage AI | rerank-2.5-lite | $0.02 | 200M tokens |
| Cohere | Rerank 4 | $2.00/1K searches | Rate-limited |
| Jina | v2-multilingual | $0.045 | 10M tokens |

---

## 3. Embeddings : état de l'art 2025

### 3.1 Voyage AI voyage-3-large (Janvier 2025)

**Source** : Voyage AI Blog
**URL** : https://blog.voyageai.com/2025/01/07/voyage-3-large/
**Date** : 7 janvier 2025
**Statut** : ⚠️ Vendor

#### Performance vs compétiteurs (nDCG@10, 100 datasets)

| Comparaison | Avantage voyage-3-large |
|-------------|-------------------------|
| vs OpenAI text-embedding-3-large (1024d) | **+10,58%** |
| vs OpenAI text-embedding-3-large (256d) | +11,47% |
| vs Cohere-v3-English | **+20,71%** |
| vs voyage-3 | +4,14% |
| vs voyage-law-2 | Supérieur (claim vendor) |

### 3.2 Validations tierces

#### MongoDB Technical Blog (2025)

**URL** : https://medium.com/mongodb/how-to-choose-the-best-embedding-model-for-your-llm-application-2f65fcdfa58d
**Statut** : ✅ Indépendant

- **Latence** : voyage-3-large **89ms/chunk** vs text-embedding-3-large 311ms
- **Finding** : "voyage-3-large produces the strongest ranking by placing the most relevant results at the top"
- **Méthodologie** : LLM-as-judge sans révéler noms modèles

#### DEV.to/DataStax (2025)

**URL** : https://dev.to/datastax/the-best-embedding-models-for-information-retrieval-in-2025-3dp5
**Statut** : ✅ Indépendant

> "The just-released Voyage-3-large is the surprise leader in embedding relevance"

### 3.3 Harvey AI + Voyage partnership

**Source** : Harvey AI Blog
**URL** : https://www.harvey.ai/blog/harvey-partners-with-voyage-to-build-custom-legal-embeddings
**Date** : Mai 2024
**Statut** : ⚠️ Marketing (deux vendors)

- Custom voyage-law-2-harvey : fine-tuned 20 billion tokens US case law
- Réduit matériel non-pertinent de ~25% vs OpenAI/Google
- Corrélation humaine ρ = 0.81-0.91

**⚠️ Note** : Ne compare pas voyage-law-2 à voyage-3-large.

### 3.4 Pricing (Janvier 2026)

| Modèle | Prix/1M | Free tier | Dimensions | Context |
|--------|---------|-----------|------------|---------|
| voyage-3-large | $0.18 | **200M** | 2048 | 32K |
| text-embedding-3-large | $0.13 | $5 crédits | 3072 | 8K |
| Cohere embed-v4 | $0.12 | Non spécifié | 1536 | **128K** |

---

## 4. Multi-query expansion

### 4.1 Études académiques 2025

#### arXiv:2501.07391 — "Enhancing RAG: Best Practices" (Janvier 2025)

**Source** : Université de Tübingen
**Statut** : ✅ Preprint académique (code public)

| Métrique | Baseline | + Query Expansion | Gain |
|----------|----------|-------------------|------|
| FActScore TruthfulQA | 53,85% | 55,82% | +1,97pp |
| + Contrastive ICL | — | **57,00%** | +3,15pp |

#### arXiv:2601.03258 — FlashRank (Janvier 2025)

| Métrique | Gain |
|----------|------|
| nDCG@10 (MS MARCO, BEIR, FinanceBench) | **+5,4%** |
| Generation accuracy | +6-8% |
| Context tokens reduction | -35% |
| Ablation : sans query expansion | **-5-6% recall** |

#### RAG-Fusion (arXiv:2402.03367, Février 2024)

| Métrique | Gain |
|----------|------|
| Accuracy réponses | +8-10% |
| Comprehensiveness (experts) | +30-40% |

### 4.2 Domaine légal spécifiquement

#### Stanford RegLab (CSLAW '25)

**URL** : https://reglab.github.io/legal-rag-benchmarks/
**Statut** : ✅ Peer-reviewed

- **Gain Recall@10** : **+10 points de pourcentage** avec query expansion structurée

#### ACM ICMR '25 — Multi-Round RAG

**URL** : https://dl.acm.org/doi/10.1145/3731715.3733451
**Statut** : ✅ Peer-reviewed

| Métrique | Single-round | Multi-round | Gain |
|----------|--------------|-------------|------|
| Recall | 57,33% | **78,67%** | +21,34pp |

### 4.3 Synthèse des gains

| Technique | Métrique | Gain | Source |
|-----------|----------|------|--------|
| Query expansion | Recall | +5-6% | FlashRank 2025 |
| RAG-Fusion | Accuracy | +8-10% | arXiv 2024 |
| Legal QE | Recall@10 | **+10pp** | Stanford 2025 |
| Multi-round Legal | Recall | **+21,34pp** | ACM 2025 |

---

## 5. Two-stage retrieval

### 5.1 Configurations documentées

#### RankRAG (Nvidia, arXiv:2407.02485)

**Statut** : ✅ Peer-reviewed

- **Recommandation** : top_k = 5-10 pour LLM final
- top-100 problématique même avec LLMs long-context
- Surpasse GPT-4 sur 9 benchmarks knowledge-intensive

#### RAG About It (2025)

**URL** : https://ragaboutit.com/adaptive-retrieval-reranking/

- Configuration enterprise : **top_k=100 → rerank 50 → output 5-10**
- Latence cible : reranking 100 docs < **300ms**
- Gain attendu : **+15-30% nDCG@5**

#### LRAGE (arXiv:2504.01840, Avril 2025)

- Configuration légale : **top 3-5 documents** par défaut
- Testé : Korean Bar Exam, LegalBench

### 5.2 Améliorations mesurées

**Source** : MyScale benchmark 2025

| Métrique | Sans reranking | Avec reranking | Gain |
|----------|----------------|----------------|------|
| Hit Rate | 0.855 | 0.895 | **+4,7%** |
| MRR | 0.640 | 0.708 | **+10,5%** |

### 5.3 Configuration recommandée

| Use Case | Initial top_k | Post-rerank | Source |
|----------|---------------|-------------|--------|
| RAG standard | 25 | 3 | Pinecone |
| Enterprise | 100 | 5-10 | RAG About It |
| **Legal RAG** | **50-100** | **10-15** | Synthèse |

---

## 6. Seuils de relevance : anti-patterns et best practices

### 6.1 Consensus académique : les seuils hardcodés sont un anti-pattern

**Source principale** : Şakar & Emekci, "Optimizing RAG Thresholds", Cambridge University, 2025
**Méthodologie** : 23 625 itérations de grid-search sur 4 domaines
**Statut** : ✅ Preprint académique

#### Résultats clés

Les seuils optimaux varient drastiquement selon le domaine :

| Domaine | Seuil optimal cosine | Impact d'un mauvais seuil |
|---------|---------------------|---------------------------|
| Financier (10Q) | 0.80 | -5,08% accuracy |
| Technique | 0.70 | -9,18% accuracy |
| Médical | 0.50 | -15,7% accuracy |
| **Juridique** | **0.70-0.75** | DRM >95% avec seuils standards |

**Finding critique** : Un seuil fixe de **0.5** laisse entre **76% et 100%** des requêtes sans contexte.

#### Citations exactes

> "Hardcoded thresholds create an illusion of control but fail when embedding models change, domains vary, or query distributions evolve."

### 6.2 Variation des scores selon les modèles d'embedding

Ni Voyage AI ni OpenAI ne fournissent de recommandations officielles de seuils. Les observations communautaires révèlent des écarts majeurs :

| Modèle | Seuil suggéré | Plage observée |
|--------|--------------|----------------|
| OpenAI ada-002 (legacy) | 0.75-0.85 | 0.50-0.95 |
| OpenAI text-embedding-3-small | **0.30-0.45** | 0.20-0.70 |
| OpenAI text-embedding-3-large | **0.27-0.40** | 0.15-0.65 |
| Voyage AI (général) | 0.35-0.50 | Similar à OpenAI v3 |
| **Contexte juridique** | **0.70-0.75** | Haute précision requise |

**Observation** : Le passage de ada-002 à text-embedding-3 induit une **baisse de ~50%** des scores. Une même requête retourne 0.79 avec ada-002, **0.33** avec text-embedding-3-small.

### 6.3 Architecture recommandée : filtrer sur le reranker, pas sur cosine

**Source** : arXiv "Beyond Component Strength", Novembre 2025
**Statut** : ✅ Preprint académique

**Finding critique** : L'adaptive thresholding seul **n'améliore rien** (40% d'abstention). C'est uniquement la **combinaison synergique** avec hybrid retrieval + reranking qui réduit l'abstention de 40% à **2%**.

#### Pipeline production recommandé

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Hybrid Retrieval (top-50 à 200)                    │
│   • Dense vector search (sémantique)                        │
│   • BM25/sparse search (mots-clés, codes, identifiants)     │
│   • Reciprocal Rank Fusion (RRF) pour combinaison           │
│   • ⚠️ PAS de seuil cosine ici                              │
├─────────────────────────────────────────────────────────────┤
│ Stage 2: Reranking (filtrer vers top-10 à 20)               │
│   • Cross-encoder reranker (Voyage, Cohere, BGE)            │
│   • ✅ Seuil sur score reranker (ex: >0.3 sur 0-1)          │
├─────────────────────────────────────────────────────────────┤
│ Stage 3: Contexte final (top-3 à 5 chunks au LLM)           │
│   • Éviter le context stuffing                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.4 Seuils reranker recommandés

Les scores de reranker sont **calibrés** contrairement aux scores cosine bruts :

| Reranker | Échelle | Seuil légal recommandé |
|----------|---------|------------------------|
| Voyage rerank-2.5 | 0-1 | **>0.3** |
| Cohere Rerank 4 | 0-1 | >0.3 |
| BGE Reranker v2 | 0-4 | >1.5-2.0 |

### 6.5 Techniques alternatives

| Technique | Impact | Complexité | Recommandation |
|-----------|--------|------------|----------------|
| **Reranker filtering** | **+20-50%** précision | Basse | ✅ **Prioritaire** |
| Percentile filtering (top 10%) | +5-10% robustesse | Basse | ✅ Simple fallback |
| Elbow detection (METEORA) | +10-20% | Haute | ⚠️ Complexe |
| Dynamic top-k | +15-30% | Haute | ⚠️ Complexe |

```python
# Percentile filtering (alternative simple)
threshold = np.percentile(scores, 90)  # Top 10%
filtered = [doc for doc in results if doc.score >= threshold]
```

### 6.6 Architectures production (Harvey, LexisNexis, Thomson Reuters)

Aucun de ces systèmes n'utilise de seuil de similarité cosine hardcodé :

| Système | Approche |
|---------|----------|
| **Harvey AI** | LanceDB + sparse/dense, reranker filtrage |
| **LexisNexis** | GraphRAG + Shepard's Knowledge Graph |
| **Thomson Reuters** | Recherche fédérée, modèles Claude par complexité |

### 6.7 Verdict et recommandation

| Pratique | Évaluation |
|----------|------------|
| Seuil cosine hardcodé (ex: 0.35) | ❌ **Anti-pattern** |
| Seuil cosine dynamique (percentile) | ⚠️ Acceptable fallback |
| **Filtrage sur score reranker** | ✅ **Best practice** |
| Hybrid + Rerank + Reranker threshold | ✅ **Production-ready** |

**Implémentation v1.9.1** : Suppression du seuil cosine `min_relevance`, filtrage uniquement via reranker `top_n` + seuil optionnel sur score reranker.

---

## 7. Contextual Retrieval : validation critique

### 7.1 Claims originaux (Anthropic, Septembre 2024)

**Source** : https://www.anthropic.com/news/contextual-retrieval

| Configuration | Réduction échecs |
|---------------|------------------|
| Contextual Embeddings seuls | -35% |
| + Contextual BM25 | -49% |
| + Reranking | **-67%** |

### 7.2 Tentatives de validation

#### arXiv:2504.19754 — "Reconstructing Context" (Avril 2025)

**Source** : Université de Bologna, ECIR 2025 Workshop
**Statut** : ✅ Académique

**Findings** :
- ✅ "Contextual retrieval preserves semantic coherence more effectively"
- ❌ **Non-réplication du -67%**
- ⚠️ "Neither technique offers a definitive solution" — trade-offs significatifs

#### LlamaIndex Implementation

**URL** : https://docs.llamaindex.ai/en/stable/examples/cookbooks/contextual_retrieval/

> "Results vary — much depends on queries, chunk size, chunk overlap, and other variables"

**Non-réplication** des pourcentages spécifiques.

### 7.3 Verdict

| Aspect | Évaluation |
|--------|------------|
| Technique valide ? | ✅ Améliore retrieval |
| -67% répliqué ? | ❌ **Non** |
| Datasets publiés ? | ❌ Non |
| Recommandation | ⚠️ Utiliser avec précaution |

### 7.4 Alternatives 2025

| Technique | Source | Avantage |
|-----------|--------|----------|
| **Late Chunking** | Jina AI (arXiv:2409.04701) | Pas d'appels LLM supplémentaires |
| LongRAG | Multiple | Exploite LLMs long-context |
| RAG-Fusion | arXiv:2402.03367 | Multi-query + RRF |

---

## 8. Citation verification

### 8.1 Techniques académiques peer-reviewed

#### CiteFix (ACL 2025 Industry Track)

**URL** : https://aclanthology.org/2025.acl-industry.23/
**Date** : Juin 2025
**Statut** : ✅ Peer-reviewed

| Métrique | Valeur |
|----------|--------|
| Amélioration relative accuracy | **+15,46%** |
| Baseline accuracy citations LLM | ~74% |
| Permet shift vers modèles | 12x moins chers, 3x plus rapides |

#### VeriCite (SIGIR-AP 2025)

**URL** : https://arxiv.org/abs/2510.11394
**Date** : Octobre 2025

- Framework 3 étapes : génération → vérification NLI → refinement
- Code public : github.com/QianHaosheng/VeriCite

#### HalluGraph (Décembre 2025)

**URL** : https://arxiv.org/pdf/2512.01659

- Framework graph-théorique pour domaine légal
- Entity Grounding + Relation Preservation
- Audit trail explicable

### 8.2 Implémentations production

| Système | Technique | Performance |
|---------|-----------|-------------|
| **Harvey** | Knowledge Source ID | >95% accuracy |
| LexisNexis | 5 checkpoints/prompt | 17% hallucinations (Stanford) |
| Westlaw | KeyCite + checker | 34% hallucinations (Stanford) |

### 8.3 Synthèse gains

| Technique | Réduction | Source |
|-----------|-----------|--------|
| RAG vs LLM général | 58-82% → 17-33% | Stanford 2025 |
| CiteFix post-processing | **+15,46% accuracy** | ACL 2025 |
| NLI verification | +23% groundedness | TechRxiv 2025 |

---

## 9. LLMs pour le légal

### 9.1 GPT-4.1-mini

**Release** : 14 avril 2025
**Context** : 1M tokens
**Pricing** : $0.40/M input, $1.60/M output

#### Benchmarks disponibles

**⚠️ Gap critique** : GPT-4.1-mini **non benchmarké** sur LegalBench (Vals.ai données incomplètes)

**LEXam Benchmark** (arXiv:2505.12864, 2025)

| Modèle | Score | MCQ Accuracy |
|--------|-------|--------------|
| Gemini-2.5-Pro | **82,2** | — |
| GPT-4.1 | 68,2 | 54,4% |
| GPT-4o | 66,2 | — |

#### Validations partenaires

| Partenaire | Claim | Statut |
|------------|-------|--------|
| Thomson Reuters | +17% multi-doc legal analysis | ⚠️ Non validé tiers |
| Blue J Tax | +53% accuracy cas fiscaux difficiles | ⚠️ Non validé tiers |

### 9.2 Leaders LegalBench (Décembre 2025)

**Source** : Vals.ai

| Rang | Modèle | Score |
|------|--------|-------|
| 1 | Gemini 3 Pro | **87,04%** |
| 2 | Gemini 3 Flash | 86,86% |
| 3 | GPT 5 | 86,02% |
| 4 | GPT 5.1 | 85,68% |

---

## 10. Architectures production

### 10.1 Harvey AI (2025)

**Sources** : Harvey Blog, VALS Report

| Composant | Choix |
|-----------|-------|
| Embeddings | Custom voyage-law-2-harvey |
| Citation verification | Knowledge Source ID (>95%) |
| Evaluation | BigLaw Bench (74% answer quality) |
| Partnership | LexisNexis (Ask LexisNexis®) |

### 10.2 Thomson Reuters CoCounsel (Août 2025)

**Source** : https://www.lawnext.com/2025/08/thomson-reuters-launches-cocounsel-legal-with-agentic-ai-and-deep-research-capabilities

- **Deep Research** : AI agentic multi-step
- KeyCite integration
- Hallucination checker intégré

### 10.3 LexisNexis Lexis+ AI

- 5 checkpoints minimum par prompt
- Shepard's® Citations Service
- **Stanford** : 17% hallucination rate (meilleur testé)
- Retrait du benchmark VALS AI — signal préoccupant

---

## 11. Analyse critique des sources

### 11.1 Matrice de fiabilité

| Source | Type | Peer-review | Conflit | Fiabilité |
|--------|------|-------------|---------|-----------|
| Stanford HAI/JELS 2025 | Journal | ✅ | ❌ | **Haute** |
| LegalBench-RAG arXiv | Preprint | ⚠️ | ❌ | **Haute** |
| CiteFix ACL 2025 | Conférence | ✅ | ❌ | **Haute** |
| Agentset Leaderboard | Benchmark | ⚠️ | ❌ | **Haute** |
| MongoDB Blog | Tutorial | ❌ | ❌ | Moyenne |
| Voyage AI Blog | Marketing | ❌ | ⚠️ | Moyenne |
| Harvey AI Blog | Marketing | ❌ | ⚠️ | **Faible** |
| Anthropic Blog | Research | ❌ | ⚠️ | Moyenne |

### 11.2 Affirmations non validées

| Affirmation | Source | Statut |
|-------------|--------|--------|
| -67% Contextual Retrieval | Anthropic | ❌ **Non répliqué** |
| 0,2% hallucinations Harvey | Harvey | ❌ **Non validé** |
| +17% legal GPT-4.1 | Thomson Reuters | ❌ **Non validé** |
| voyage-3-large > voyage-law-2 | Voyage AI | ⚠️ **Pas de test tiers** |

### 11.3 Gaps critiques identifiés

| Domaine | Gap | Priorité |
|---------|-----|----------|
| Reranking légal | Aucun benchmark 3 modèles sur mêmes données légales | 🔴 |
| GPT-4.1-mini | Non benchmarké LegalBench | 🔴 |
| voyage-law-2 vs v3-large | Aucune validation tierce | 🔴 |
| Contextual Retrieval | -67% non répliqué | 🟡 |

---

## 12. Références complètes

### Peer-reviewed (priorité maximale)

1. Magesh, V. et al. "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools." *JELS*, Vol. 22(2), 2025.
   https://onlinelibrary.wiley.com/doi/full/10.1111/jels.12413

2. "CiteFix: Enhancing RAG Accuracy Through Post-Processing Citation Correction." *ACL 2025 Industry Track*.
   https://aclanthology.org/2025.acl-industry.23/

3. "LegalBench-RAG: A Benchmark for RAG in the Legal Domain." arXiv:2408.10343, 2024.
   https://arxiv.org/abs/2408.10343

4. "Enhancing RAG: A Study of Best Practices." arXiv:2501.07391, Janvier 2025.
   https://arxiv.org/abs/2501.07391

5. "Two-Stage Retrieval: FlashRank Reranking and Query Expansion." arXiv:2601.03258, Janvier 2025.
   https://arxiv.org/abs/2601.03258

6. "RankRAG: Unifying Context Ranking with RAG in LLMs." arXiv:2407.02485, Juillet 2024.
   https://arxiv.org/abs/2407.02485

7. "VeriCite: Towards Reliable Citations in RAG." SIGIR-AP 2025.
   https://arxiv.org/abs/2510.11394

8. "HalluGraph: Auditable Hallucination Detection for Legal RAG." arXiv:2512.01659, Décembre 2025.
   https://arxiv.org/pdf/2512.01659

9. "LEXam: Benchmarking Legal Reasoning on 340 Law Exams." arXiv:2505.12864, 2025.
   https://arxiv.org/abs/2505.12864

10. "Multi-Round RAG for Legal Document Analysis." ACM ICMR '25.
    https://dl.acm.org/doi/10.1145/3731715.3733451

11. Şakar & Emekci. "Optimizing RAG Thresholds." Cambridge University, 2025.
    Grid-search sur 23 625 itérations.

12. Krishnan. "Beyond Component Strength: Synergistic Integration and Adaptive Calibration in Multi-Agent RAG Systems." arXiv, Novembre 2025.
    https://arxiv.org/abs/2511.21729

### Benchmarks indépendants

13. Agentset. "Reranker Leaderboard." Novembre 2025.
    https://agentset.ai/rerankers

12. VALS AI. "Legal AI Report." Février 2025.
    https://www.vals.ai/vlair

13. Stanford RegLab. "Legal RAG Benchmarks." CSLAW '25.
    https://reglab.github.io/legal-rag-benchmarks/

14. Charlotin, D. "AI Hallucination Cases Database."
    https://www.damiencharlotin.com/hallucinations/

### Évaluations tierces

15. MongoDB. "How to Choose the Best Embedding Model." 2025.
    https://medium.com/mongodb/how-to-choose-the-best-embedding-model-for-your-llm-application-2f65fcdfa58d

16. DEV.to/DataStax. "Best Embedding Models 2025."
    https://dev.to/datastax/the-best-embedding-models-for-information-retrieval-in-2025-3dp5

17. RAG About It. "Adaptive Retrieval Reranking." 2025.
    https://ragaboutit.com/adaptive-retrieval-reranking/

### Vendor (utiliser avec précaution)

18. Voyage AI. "voyage-3-large." Janvier 2025.
    https://blog.voyageai.com/2025/01/07/voyage-3-large/

19. Voyage AI. "Pricing." Décembre 2025.
    https://docs.voyageai.com/docs/pricing

20. Anthropic. "Contextual Retrieval." Septembre 2024.
    https://www.anthropic.com/news/contextual-retrieval

21. Harvey AI. "BigLaw Bench: Hallucinations." Octobre 2024.
    https://www.harvey.ai/blog/biglaw-bench-hallucinations

22. Harvey AI. "Voyage Partnership." Mai 2024.
    https://www.harvey.ai/blog/harvey-partners-with-voyage-to-build-custom-legal-embeddings

23. OpenAI. "GPT-4.1 Release." Avril 2025.
    https://openai.com/index/gpt-4-1/

### Implémentations

24. LlamaIndex. "Contextual Retrieval Cookbook."
    https://docs.llamaindex.ai/en/stable/examples/cookbooks/contextual_retrieval/

25. Thomson Reuters. "CoCounsel Legal." Août 2025.
    https://www.lawnext.com/2025/08/thomson-reuters-launches-cocounsel-legal-with-agentic-ai-and-deep-research-capabilities

---

*Document de recherche — Legal RAG PoC v1.9*
*Dernière mise à jour : 11 janvier 2026*
*Prochaine révision recommandée : Avril 2026 (évolution rapide du domaine)*

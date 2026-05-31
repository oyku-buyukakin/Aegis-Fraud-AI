# Aegis Fraud AI — Technical Architecture

## Overview

Aegis is a multi-layer fraud detection system that combines:

- **Classical ML pipeline** for feature engineering and anomaly scoring
- **Rule-based engine** with YAML-defined rules and conflict resolution
- **RAG knowledge base** for grounded fraud-policy reasoning
- **Multi-agent orchestration** for parallel signal processing
- **FastAPI** for production-ready endpoint exposure

---

## System Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI (Layer 5)                           │
│  / (dashboard)  /score  /explain  /rules/evaluate              │
│  /rules/list  /rag/query  /health                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│              Multi-Agent Orchestration (Layer 4)                │
│  FraudOrchestrator → [AmountTime | Identity | Velocity]        │
│                    → AnomalyAgent → DecisionAgent              │
│                    → ExplanationAgent (local LLM via Ollama)   │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│              RAG Knowledge Base (Layer 3)                       │
│  docs/knowledge_base → chunk → embed → FAISS → retrieve        │
│  Local LLM (Ollama / llama3.2) for grounded answer generation  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│          Rule Engine + Context Intelligence (Layer 2)           │
│  fraud_rules.yaml → rule evaluation → conflict resolution      │
│  context_adjuster → final anomaly score adjustment             │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│            ML Anomaly & Score Fusion (Layer 1)                  │
│  IsolationForest + XGBoost + LightGBM + mRMR feature selection │
│  Ensemble fusion → adjusted_anomaly_score                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│           Data & Feature Engineering (Layer 0)                  │
│  IEEE-CIS CSV → merge → impute → signal factory                │
│  time signals, entity signals, relationship signals            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Notebook Pipeline

### NB 01 — Signal Discovery EDA
- Loads raw IEEE-CIS `train_transaction.csv` + `train_identity.csv`
- Exploratory: missing value analysis, class imbalance, correlation patterns
- Identifies fraud-predictive features

### NB 02 — Schema Radar Profiling
- Reads `train_merged.pkl`, `test_merged.pkl`
- Statistical profiling with IsolationForest and sklearn pipelines
- Validates schema consistency between train and test splits

### NB 03 — Fraud Signal Factory
- Derives engineered features from raw columns:
  - Time signals: `is_night_transaction`, `is_business_hour`
  - Entity signals: `entity_trusted_score`, `is_high_risk_entity`
  - Relationship signals: amount deviations, velocity checks
- Output: `train/test_feature_engineered.pkl`

### NB 04 — Anomaly Risk Engine
- IsolationForest on engineered features → `anomaly_score`
- Multi-detector ensemble for entity, temporal, multivariate, and field-drift signals
- Output: `train/test_anomaly_scored.pkl`

### NB 05 — Risk Score Fusion
- XGBoost + LightGBM trained on anomaly features
- mRMR feature selection for interpretability
- Cross-validation + ensemble weighting
- Outputs: `train/test_score_fused.pkl`, `reports/score_fusion_metrics.json`

### NB 06 — Context-Aware Adjustment
- LightGBM context model: product/time/entity context signals
- Adjusts raw score → `adjusted_anomaly_score` with context awareness
- Reduces false positives on legitimate high-value transactions
- Output: `train/test_context_adjusted.pkl`

### NB 07 — Explainable Rule Engine
- Loads `configs/fraud_rules.yaml` (10 rules: RULE_001–RULE_010)
- Evaluates rules on `adjusted_anomaly_score` + context features
- Conflict resolution via `configs/conflict_resolution.yaml`:
  - Action: `highest_priority`
  - Severity: `highest_severity`
  - Max rules per transaction: 5
- Generates human-readable `explanation_template` per rule

### NB 08 — Fraud Knowledge Base (RAG)
- Loads 5 knowledge base documents from `docs/knowledge_base/`
- Chunks → embeds with `paraphrase-MiniLM-L3-v2` (or hashing fallback)
- FAISS inner-product index for cosine similarity search
- Constructs RAG prompt → Ollama (`llama3.2`) for grounded answers
- Embedding cache: `data/interim/kb_embeddings_*.npy`, `kb_faiss_*.index`

### NB 09 — Multi-Agent Orchestration
- **Parallel stage 1 (concurrent):**
  - `AmountTimeSignalAgent`: amount + time features
  - `IdentitySignalAgent`: email domain + device trust + country
  - `VelocitySignalAgent`: account age + velocity
- **Sequential stages:**
  - `AnomalyAgent`: weighted score from signals
  - `DecisionAgent`: threshold-based action (APPROVE / FLAG_REVIEW / BLOCK)
  - `ExplanationAgent`: Ollama `llama3.2` LLM explanation
- Communication via `AgentMessage` + shared `Blackboard`
- `ThreadPoolExecutor` for parallel signal agents

### NB 10 — API Development
- **Modular FastAPI app** under `src/api/`: `schemas.py`, `services/`, thin `routes/`,
  `containers.py` (``dependency_injector``), `main.py` app factory
- **Dependency Injection:** singleton services wired via `Container`; routes receive
  `ScoringService`, `ExplanationService`, `RuleService`, `RagService` through
  `Depends(Provide[Container....])` — satisfies the bonus DI/container requirement
- **Scoring strategy:** `ScoringService` uses an `AnomalyModel` protocol. When
  `models/anomaly_model.joblib` exists (exported by `scripts/export_scoring_model.py`
  from NB03–06 pipeline data), `/score` uses a trained LightGBM on API-compatible
  signals; otherwise it falls back to the transparent signal-weight heuristic
- Self-explanatory dashboard at `/` (HTML in `src/api/dashboard.py`)
- Notebook 10 is a **thin integration layer** — imports `create_app()` from `src`,
  starts uvicorn, runs httpx tests (no duplicated business logic)
- Auto-generated Swagger UI also available at `/docs`

---

## API Endpoints

### `POST /score`

Computes a fraud risk score via the injected `ScoringService`. When
`models/anomaly_model.joblib` is present, a LightGBM trained on API-compatible
pipeline signals is used (`predict_proba`); otherwise a weighted-signal heuristic
is used as a transparent fallback.

**Request:**
```json
{
  "transaction_id": "TXN-001",
  "TransactionAmt": 875.00,
  "hour": 2,
  "account_age_days": 1,
  "num_txn_last_1h": 9,
  "P_emaildomain": "protonmail.com",
  "card_type": "credit",
  "country_mismatch": 1
}
```

> `device_trust_score` is optional (defaults to `0.5`); external callers are not
> expected to know it, so the dashboard does not collect it.

**Response:**
```json
{
  "transaction_id": "TXN-001",
  "anomaly_score": 0.75,
  "risk_level": "HIGH",
  "triggered_signals": ["is_night_transaction", "is_new_account", "is_high_amount", ...]
}
```

### `POST /explain`

Same input as `/score` plus optional `anomaly_score`. Returns a plain-language,
sentence-form explanation of why the transaction was flagged.

### `GET /rules/list`

Returns the catalog of fraud rules (`rule_id`, `name`, `severity`, `action`,
plain-language `description`). Powers the dashboard's "which fraud rules can flag a
transaction?" panel without requiring any input.

### `POST /rules/evaluate`

Accepts a `features` dict matching fields in `fraud_rules.yaml`.

**Response includes:**
- `fired_rules`: list of matched rules with explanation
- `final_action`: highest-priority action
- `final_severity`: highest severity across fired rules

### `POST /rag/query`

```json
{ "query": "What happens when a new account makes a high-value transaction?", "top_k": 3 }
```

Returns ranked knowledge base chunks by a stopword-aware term-overlap score. Raw data
dumps (JSON exports, synthetic-case logs) are excluded, and matches below
`_MIN_RAG_SCORE` (0.8) are dropped so weak/irrelevant queries return no results
instead of a misleading answer.

### `GET /health`

Returns `status`, `rules_loaded`, `kb_chunks`, and the active scoring `model`
(`pipeline-lightgbm-api-signals` or `signal-weight-heuristic`).

---

## Rule Engine Schema

Rules are defined in `configs/fraud_rules.yaml`:

```yaml
- id: RULE_001
  name: "Extreme Composite Anomaly Risk"
  priority: 1
  severity: CRITICAL
  logic: AND
  conditions:
    - field: adjusted_anomaly_score
      operator: gte
      value: 0.78
    - field: anomaly_high_flag_count
      operator: gte
      value: 10
  action: FLAG_CRITICAL_REVIEW
  explanation_template: >
    Adjusted anomaly score is {adjusted_anomaly_score:.3f} and
    {anomaly_high_flag_count} high anomaly flags fired.
```

Supported operators: `gte`, `gt`, `lte`, `lt`, `eq`, `neq`

---

## Data Flow

```
train_transaction.csv ─┐
train_identity.csv    ─┼─► merge ─► feature_engineered.pkl
test_transaction.csv  ─┤              │
test_identity.csv     ─┘              ▼
                                anomaly_scored.pkl
                                      │
                                      ▼
                                score_fused.pkl
                                      │
                                      ▼
                               context_adjusted.pkl
                                      │
                          ┌───────────┼───────────┐
                          ▼           ▼           ▼
                     Rule engine   RAG KB    Multi-agent
                     (NB 07)       (NB 08)   (NB 09)
                          │           │           │
                          └───────────┴───────────┘
                                      │
                                      ▼
                                  FastAPI
                                  (NB 10)
```

---

## Local LLM Integration

Both NB 08 and NB 09 use Ollama for local inference:

- **Model:** `llama3.2` (2 GB, recommended for Mac M1)
- **Context window:** 1536–2048 tokens (kept small for memory)
- **Threads:** 2 (safe for 8 GB unified memory)
- **Temperature:** 0.1 (deterministic, fact-grounded responses)

To start:
```bash
ollama serve          # terminal 1
ollama pull llama3.2  # terminal 2 (first time only)
```

---

## Performance Notes (Mac M1)

| Component | Memory | Notes |
|-----------|--------|-------|
| Notebook pipeline (NB 01–07) | ~4 GB peak | Pandas + sklearn on IEEE-CIS data |
| Embedding generation (NB 08) | ~500 MB | `paraphrase-MiniLM-L3-v2`, batch_size=2 |
| FAISS index (NB 08) | <10 MB | 84 chunks × 384 dims |
| Ollama llama3.2 | ~2.5 GB | Quantized model |
| FastAPI server (NB 10) | <100 MB | Pure Python, no ML models in API |

Embedding results are cached to `data/interim/` after the first run so re-running NB 08 is instant.

---

## Config Files

| File | Purpose |
|------|---------|
| `configs/fraud_rules.yaml` | 10 rule definitions with conditions, actions, severities |
| `configs/conflict_resolution.yaml` | Multi-rule conflict resolution policy |

---

## Source Code

| File | Responsibility |
|------|---------------|
| `src/api/main.py` | App factory, DI container bootstrap, `/` dashboard, `/health` |
| `src/api/containers.py` | `dependency_injector` Container — singleton service graph |
| `src/api/schemas.py` | Shared Pydantic request/response models |
| `src/api/services/scoring_service.py` | `AnomalyModel` strategy + `ScoringService` |
| `src/api/services/explanation_service.py` | Plain-language explainability |
| `src/api/services/rule_service.py` | Rule evaluation + catalog |
| `src/api/services/rag_service.py` | Stopword/threshold-aware KB retrieval |
| `src/api/routes/*.py` | Thin HTTP handlers (`@inject` + `Depends`) |
| `src/api/dashboard.py` | Self-explanatory dashboard HTML served at `/` |
| `scripts/launch_api.py` | Production entry point (`uvicorn src.api.main:app`) |
| `scripts/export_scoring_model.py` | Train + export `models/anomaly_model.joblib` |
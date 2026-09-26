# MedSignal — Evaluation Log

A running record of evaluation runs. Copy the **Run entry** template for each
new run and fill it in. Metrics come from `eval/eval_retrieval.py` (planning &
retrieval) and `eval/eval_agent.py` (end-to-end agent).

---

## Environment

| Field | Value |
|---|---|
| Date | _YYYY-MM-DD_ |
| Machine | _dev / personal laptop_ |
| Commit | _git sha_ |
| LLM | _llama-3.1-70b-versatile / fallback / none_ |
| Embedding model | _BioLORD-2023-C_ |
| Graph backend | _networkx / neo4j_ |
| Data | _sample / real_ (FAERS __, PubMed __, trials __, labels __) |

---

## Dataset

- Test set: `eval/test_queries.json` — **20** queries (7 easy / 7 medium / 6 hard).
- Scoring keys per query: `expected_intent`, `expected_drugs`,
  `expected_evidence_grade_min`, `expected_sources`, `difficulty`.

---

## Run entry template

### Run _N_ — _YYYY-MM-DD_

**Config:** _model, data scope, notable changes since last run._

#### Retrieval metrics (`eval_retrieval.py`)

| Metric | Value | Target | Notes |
|---|---|---|---|
| Intent accuracy | __% | ≥ 85% | |
| Drug detection | __% | ≥ 90% | |
| Grade pass rate (actual ≥ expected_min) | __% | ≥ 70% | requires API |
| Source coverage (expected ⊆ consulted) | __% | ≥ 75% | requires API |

By difficulty:

| Difficulty | Intent | Drugs | Grade | Sources |
|---|---|---|---|---|
| easy | __% | __% | __% | __% |
| medium | __% | __% | __% | __% |
| hard | __% | __% | __% | __% |

#### Agent metrics (`eval_agent.py`)

| Metric | Value | Target | Notes |
|---|---|---|---|
| Completed (no error) | _/20_ | 20/20 | |
| Plan quality (1–4 valid sub-queries) | __% | ≥ 90% | |
| Citation coverage (non-E answers cited) | __% | 100% | every claim cited |
| Avg iterations (fewer is better) | __ | ≤ 2.0 | max 3 |
| Contradiction detection (flagged where expected) | __/__ | — | e.g. label-vs-FAERS |

#### Failure analysis

- _Query → what went wrong → suspected cause → fix / follow-up._

#### Notes & decisions

- _Prompt/version changes, threshold tweaks, data gaps, next actions._

---

## History

| Run | Date | Intent | Drugs | Grade | Plan | Citations | Avg iters |
|-----|------|--------|-------|-------|------|-----------|-----------|
| 1 | | | | | | | |

# MedSignal

> **Multi-source agentic RAG for drug safety intelligence — because scattered safety data costs lives.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/agent-LangGraph-1C3C3C.svg)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/vectors-Qdrant-DC244C.svg)](https://qdrant.tech/)
[![Neo4j](https://img.shields.io/badge/graph-Neo4j%20%2F%20NetworkX-018BFF.svg)](https://neo4j.com/)

---

## The problem

When a clinician asks *"is this drug safe for my patient?"*, the honest answer is buried across four disconnected public systems. The **FDA's FAERS** database holds millions of spontaneous adverse-event reports. **PubMed** has the peer-reviewed literature. **ClinicalTrials.gov** has the controlled-trial safety data. **DailyMed / drug labels** hold the regulator-approved truth. None of them talk to each other.

So a pharmacovigilance analyst does it by hand — pulling FAERS counts, searching PubMed, cross-referencing trial arms, and reading label sections — spending hours or days to answer a single question, and repeating it for the next drug. Signals that live *across* sources (a rising FAERS count for a reaction that isn't yet on the label) are exactly the ones most likely to be missed, because no single source shows them.

The tools that do unify this data are enterprise pharma platforms: expensive, closed, and built before modern LLMs. **MedSignal** is a demonstration that the same job can be done with an open stack and *agentic* retrieval — an AI that plans which sources to query, retrieves across all four, weighs them by authority, and returns a **graded, cited** answer instead of a wall of links. It's a portfolio project, but the architecture is real: stateless API, a biomedical knowledge graph, and evidence grading you can audit.

---

## Key features

- **Multi-source ingestion** — FAERS, PubMed, ClinicalTrials.gov, and DailyMed labels, each with its own loader that parses both the raw public-API format and a simplified sample format.
- **Biomedical knowledge graph** — drugs, adverse events, conditions, trials, publications, and drug classes connected by typed edges (`REPORTED_WITH`, `INTERACTS_WITH`, `SAME_CLASS_AS`, …). Runs on NetworkX in dev, Neo4j in prod, behind one interface.
- **Agentic RAG with query planning (LangGraph)** — not a retrieve-and-generate pipeline. The agent classifies intent, decomposes the query into source-targeted sub-queries, retrieves, **evaluates sufficiency**, and iterates (up to 3 loops) before synthesizing.
- **Evidence grading (A–E)** — every answer carries a grade based on source authority and cross-source convergence. A drug-label confirmation outranks a stack of individual FAERS reports.
- **Signal detection** — surfaces drug/adverse-event pairs with high FAERS counts that **aren't yet reflected in the label** — the emerging-signal case that matters most.
- **Production-shaped** — a stateless, testable **FastAPI** backend (auto-generated OpenAPI docs) with a **Streamlit** UI that calls it over HTTP. The API is the product; the UI is a client.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                  │
│                 Streamlit App (calls FastAPI backend)                  │
│   [Query] [Drug Explorer] [Signal Detection] [Graph] [Analytics]       │
└───────────────────────────────┬────────────────────────────────────── ┘
                                 │ HTTP
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                            FastAPI BACKEND                             │
│   /api/query   /api/drugs   /api/graph   /api/ingest   /api/analytics  │
└───────────────────────────────┬────────────────────────────────────── ┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     AGENTIC RAG ENGINE (LangGraph)                     │
│    Query Planner  ─▶  Retrieval Loop  ─▶  Evidence Synthesizer         │
│    (classify /        (search /            (merge / grade /            │
│     decompose /        evaluate /           cite)                      │
│     pick sources)      iterate)                                        │
│                             │                                          │
│                RETRIEVAL LAYER + Re-ranker                             │
│      FAERS · PubMed · Trials · Drug Labels · Knowledge Graph           │
└───────────────────────────────┬────────────────────────────────────── ┘
              ┌──────────────────┼────────────────────┐
              ▼                  ▼                     ▼
     ┌───────────────┐  ┌─────────────────┐  ┌──────────────────┐
     │  Vector Store │  │ Knowledge Graph │  │    SQL Store     │
     │   (Qdrant)    │  │ (NetworkX/Neo4j)│  │    (SQLite)      │
     │ 4 collections │  │  drugs, AEs,    │  │  reports, labels,│
     │  per source   │  │  trials, pubs   │  │  trials, queries │
     └───────────────┘  └─────────────────┘  └──────────────────┘
              ▲                  ▲                     ▲
              └──────────────────┼────────────────────┘
┌──────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINES                             │
│   FAERS · PubMed · Clinical Trials · Drug Labels                       │
│   fetch → parse → (LLM extract) → chunk → embed → store + graph        │
└──────────────────────────────────────────────────────────────────────┘
```

### Agent state machine

```
        START
          │
          ▼
   ┌──────────────┐
   │ Query Planner│  classify intent · decompose · pick sources
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ Source Router│  dispatch to the sub-query's targeted retrievers
   └──────┬───────┘
   ┌──────┼───────────────┬───────────────┐
   ▼      ▼               ▼               ▼
 FAERS  PubMed         Trials /        Knowledge
 Retr.  Retr.          Labels Retr.    Graph
   └──────┴───────┬───────┴───────────────┘
                  ▼
        ┌───────────────────┐
        │ Evidence Evaluator│  sufficient? contradictions? need more?
        └─────────┬─────────┘
             ┌────┴─────┐
        sufficient   insufficient ──▶ back to Source Router (max 3 loops)
             │
             ▼
        ┌───────────────────┐
        │Evidence Synthesizer│  merge · grade (A–E) · cite
        └─────────┬─────────┘
                  ▼
                 END
```

The state threaded through the graph accumulates evidence, source results, and contradictions across iterations (LangGraph reducers merge parallel updates), and ends with a graded, cited `AgentResult`.

---

## Evidence grading

Every answer is graded on source authority **and** cross-source convergence:

| Grade | Label        | Criteria                                                   |
|:-----:|--------------|------------------------------------------------------------|
| **A** | Strong       | Confirmed in a drug label **and** supported by trial data  |
| **B** | Moderate     | Many FAERS reports (>50) **+** supporting literature (2+)  |
| **C** | Suggestive   | Literature case reports **+** a FAERS signal (10+ reports) |
| **D** | Weak         | Isolated FAERS reports only (<10), or a single case report |
| **E** | Insufficient | No evidence found in any source                            |

**Source authority ranking:** Drug Label > Clinical Trial > Systematic Review > Case Report > Individual FAERS Report.

---

## Tech stack

| Component        | Technology                              | Why                                                    |
|------------------|-----------------------------------------|--------------------------------------------------------|
| LLM              | Groq — `llama-3.1-70b-versatile`        | Strong reasoning for planning/synthesis; fast + free tier |
| Embeddings       | BioLORD-2023-C (sentence-transformers)  | Biomedical-domain vectors, far better than generic     |
| Vector store     | Qdrant (local/persistent)               | Payload filtering + per-source collections             |
| Knowledge graph  | NetworkX (dev) / Neo4j (prod)           | Real graph traversals behind one interface             |
| SQL store        | SQLite                                  | Structured records, query log, analytics               |
| Agent framework  | LangGraph                               | State machine for iterative agentic retrieval          |
| Backend          | FastAPI                                 | Stateless, testable, async, auto OpenAPI docs          |
| Frontend         | Streamlit + Plotly + Pyvis              | Multipage demo UI over the API                         |
| Validation       | Pydantic v2                             | Typed schemas end-to-end — no raw dicts flowing through |

---

## Quick start

**Prerequisites:** Python 3.11+, and (for real answers) a free [Groq API key](https://console.groq.com/keys).

```bash
# 1. Clone & install
git clone <your-fork-url> medsignal && cd medsignal
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure (add your GROQ_API_KEY)
cp .env.example .env

# 3. Generate offline sample data (no network needed)
python scripts/seed_sample_data.py

# 4. Ingest into the stores + knowledge graph
python scripts/ingest_all.py

# 5. Run the backend (terminal 1)
uvicorn api.main:app --port 8000            # docs at http://localhost:8000/docs

# 6. Run the UI (terminal 2)
streamlit run app/streamlit_app.py          # http://localhost:8501
```

Prefer the terminal? `python scripts/query_cli.py` gives you an interactive, cited Q&A loop.

> **Note:** ingestion downloads the BioLORD embedding model on first run, and full answers call the Groq API — both need outbound network access.

---

## Usage examples

| Ask | What MedSignal does |
|-----|---------------------|
| *"Is metformin linked to lactic acidosis?"* | Plans FAERS + label + literature retrieval; aggregates FAERS report counts from the graph, confirms against the label warning, cites both → **Grade B/C**. |
| *"What drug interactions should I watch for with warfarin?"* | Pulls `INTERACTS_WITH` edges (aspirin, ibuprofen, amiodarone) with label severities and supporting literature → **Grade A**. |
| *"Compare the GI safety of ibuprofen vs aspirin."* | Detects a comparison intent and both drugs; retrieves each drug's GI adverse-event profile across FAERS + labels and contrasts them. |
| *"Are there emerging signals for the statins not yet in their labels?"* | Signal detection: finds high-FAERS drug/AE pairs whose event isn't in the label's warnings/reactions. |
| *"A 72-year-old on warfarin was prescribed amiodarone — what monitoring is needed?"* | Multi-step: interaction edge (major) + FAERS bleeding signal + label guidance → INR-monitoring answer at **Grade A**. |

Each response includes the answer, an A–E grade, expandable citations (source icon, reference ID, snippet), and the agent trace (plan, sub-queries, sources consulted, iterations).

---

## Data sources

| Source | Provides | Access | Sample volume |
|--------|----------|--------|---------------|
| **FDA FAERS** | Spontaneous adverse-event reports: patient, drugs + roles, MedDRA reactions, outcomes | openFDA `drug/event.json` | 50 reports |
| **PubMed** | Abstracts on drug-safety topics: title, abstract, authors, MeSH, study type | NCBI E-utilities | 30 articles |
| **ClinicalTrials.gov** | Trials: phase, status, interventions, adverse-event results | CTgov REST API v2 | 20 trials |
| **DailyMed / labels** | Regulator-approved labels: indications, contraindications, warnings, interactions, boxed warnings | openFDA `drug/label.json` | 15 labels |

The committed **sample** data (`scripts/seed_sample_data.py`) is fully offline and pre-wired so the graph has meaningful connections. **Real** data is fetched on a machine with network access via `scripts/fetch_*.py`, which write into `data/raw/<source>/`; then `python scripts/ingest_all.py --data-dir data/raw`.

---

## How it works

**Ingestion (per source).** A loader parses the source (raw API format *or* sample format) into validated Pydantic models → records are written to SQLite → **knowledge-graph** nodes and typed edges are built (FAERS `REPORTED_WITH` counts are aggregated; labels add `INTERACTS_WITH` / `CONTRAINDICATED_FOR`; classes add `SAME_CLASS_AS`) → text is **chunked** per source → **embedded** with BioLORD → upserted into the source's Qdrant collection. Every step is resilient: one bad record is logged and skipped, never aborting the batch.

**Query (agent).** `plan → retrieve → evaluate → synthesize`. The planner classifies intent (safety profile, interaction, adverse-event lookup, signal detection, comparison, general) and emits 1–4 source-targeted sub-queries. Retrievers fuse **vector** search with **structured** SQL/graph lookups. The evaluator checks whether evidence is sufficient (≥ N pieces across ≥ 2 sources) and loops if not. The synthesizer re-ranks by authority, grades A–E, and produces a cited answer. With no Groq key, it degrades gracefully to a keyword planner + deterministic synthesis.

---

## API documentation

Interactive OpenAPI docs are auto-generated at **`http://localhost:8000/docs`**.

| Method & path | Purpose |
|---|---|
| `GET /` | Health check |
| `POST /api/query` | Run the agent over a question → graded, cited answer |
| `GET /api/drugs` | List drugs in the graph |
| `GET /api/drugs/{name}` | Safety profile (graph) + label (SQL) |
| `GET /api/drugs/{name}/interactions` | Interacting drugs with severities |
| `GET /api/drugs/{name}/adverse-events` | Adverse events with report counts |
| `GET /api/graph/stats` | Node/edge counts, top-connected drugs |
| `GET /api/graph/neighbors/{node_id}` | Directly connected nodes |
| `GET /api/graph/path/{drug1}/{drug2}` | Shortest path between two drugs |
| `POST /api/ingest` · `GET /api/ingest/status` | Trigger ingestion · store counts |
| `GET /api/analytics/{top-drugs,top-adverse-events,outcome-distribution,drug-class-profile,signals,query-stats}` | Aggregate views |

---

## Evaluation

A 20-query gold set (`eval/test_queries.json`, 7 easy / 7 medium / 6 hard) with expected intent, drugs, minimum evidence grade, and sources.

```bash
python eval/eval_retrieval.py    # intent accuracy, drug detection, grade pass rate, source coverage
python eval/eval_agent.py        # plan quality, iteration efficiency, citation coverage, contradictions
```

Both print Rich tables and save JSON results; record each run in **`eval/EVAL_LOG.md`** (environment, per-difficulty breakdown, failure analysis) so quality is tracked over time, not asserted once. `eval_retrieval.py` runs offline (keyword planner) and adds grade/source scoring when the API is up.

---

## Project structure

```
medsignal/
├── api/                    # FastAPI backend
│   ├── main.py             #   app + CORS + health
│   ├── dependencies.py     #   cached store/agent providers
│   └── routes/             #   query, drugs, graph, ingest, analytics
├── app/                    # Streamlit UI
│   ├── streamlit_app.py    #   home + sidebar status
│   ├── pages/              #   query, drug explorer, signals, graph, analytics
│   └── components/         #   api client, theme, cards, graph viz
├── src/
│   ├── models/             # Pydantic schemas + enums
│   ├── ingestion/          # loaders, chunker, extractor, pipeline
│   ├── storage/            # vector (Qdrant), graph (NetworkX/Neo4j), SQL
│   ├── retrieval/          # per-source retrievers, planner, reranker, grader
│   ├── agents/             # LangGraph state, nodes, edges, graph
│   └── utils/              # embeddings, Groq client, medical terms
├── scripts/                # seed, fetch_*, ingest_all, reset_stores, query_cli, test_e2e
├── eval/                   # test_queries.json, eval_retrieval, eval_agent, EVAL_LOG
├── tests/                  # pytest suite (offline: pytest -m "not slow")
└── data/                   # sample/ (committed), raw/, qdrant/, sqlite, graph json
```

---

## Development

**Run the offline test suite** (no API, no model downloads):

```bash
pytest -m "not slow"        # full suite runs on the personal laptop
```

**Add a new data source.** Add a Pydantic schema in `src/models/schemas.py`, a loader in `src/ingestion/` (parse → validated models), a `chunk_*` function in `chunker.py`, a Qdrant collection + `DataSource` enum value, then wire an `ingest_*` method in `pipeline.py` and (optionally) a retriever in `src/retrieval/`.

**Swap the graph backend to Neo4j.** Set `GRAPH_BACKEND=neo4j` and the `NEO4J_*` variables in `.env`. `get_graph_store()` returns the `Neo4jGraphStore` (full Cypher implementation) instead of `NetworkXGraphStore` — no other code changes; both satisfy the same `GraphStoreProtocol`.

---

## Future enhancements

- **Real-time FAERS streaming** — incremental ingestion of new quarterly extracts / openFDA deltas.
- **MedDRA hierarchy integration** — roll adverse events up to System Organ Class for aggregation and de-duplication.
- **UMLS concept linking** — normalize drugs and events to CUIs for cross-vocabulary matching.
- **Multi-language support** — non-English labels and literature.
- **Clinical decision support integration** — surface graded signals inside EHR prescribing workflows.
- **FDA label-change monitoring** — alert when an emerging signal finally lands in a label, closing the detection loop.

---

## License

Released under the **MIT License** — see [LICENSE](LICENSE).

---

*MedSignal is a portfolio/research demonstration. It is **not** a medical device and must not be used for clinical decision-making. Always consult primary sources and a qualified professional.*

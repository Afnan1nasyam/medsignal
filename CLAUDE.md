# CLAUDE.md — MedSignal

## Project Overview
A multi-source agentic RAG system for drug safety intelligence. Ingests data from 4 public sources (FDA FAERS, PubMed, ClinicalTrials.gov, DailyMed drug labels), builds a biomedical knowledge graph, and uses a LangGraph-based agent for multi-step retrieval with evidence grading.

## ⚠️ CRITICAL: DO NOT MAKE API CALLS ⚠️
This dev machine has Zscaler corporate proxy that blocks ALL outbound API calls. The following will FAIL and you must NEVER attempt them during development:
- ❌ Do NOT call the Groq API (groq.Groq().chat.completions.create) — it will timeout/fail
- ❌ Do NOT call the openFDA API (api.fda.gov) — blocked
- ❌ Do NOT call the PubMed/NCBI API (eutils.ncbi.nlm.nih.gov) — blocked
- ❌ Do NOT call the ClinicalTrials.gov API — blocked
- ❌ Do NOT download models from HuggingFace (SentenceTransformer('...') auto-downloads) — blocked
- ❌ Do NOT run scripts/fetch_*.py — those are for the personal laptop only
- ❌ Do NOT run scripts/test_e2e.py — needs Groq API
- ❌ Do NOT run scripts/ingest_all.py with --enrich flag — needs Groq API
- ❌ Do NOT run pytest tests marked @pytest.mark.slow — they need Groq API

If you need to verify code that uses GroqClient or EmbeddingModel, use SYNTAX CHECKS ONLY:
python -c "import ast; ast.parse(open('file.py').read()); print('Syntax OK')"

## What you CAN and SHOULD do
- ✅ Run `pip install -r requirements.txt` (PyPI access works through Zscaler)
- ✅ Run Python scripts, import modules, verify code works
- ✅ Run `pytest tests/ -v -m "not slow"` — local tests without API calls
- ✅ Run `python scripts/seed_sample_data.py` — writes hardcoded JSON, no network
- ✅ Verify imports, config loading, chunker logic, graph operations, SQLite, Qdrant local mode
- ✅ Run `uvicorn api.main:app` to verify FastAPI starts and /docs renders
- ✅ Run `streamlit run app/streamlit_app.py` to verify UI layout renders
- ✅ Run FastAPI TestClient tests (they don't make real API calls)
- ✅ Test NetworkX graph store, SQLite store, Qdrant local mode — all offline

## Test Marking Rule
- Any test that calls GroqClient.generate(), GroqClient.generate_json(), or loads a sentence-transformers model MUST be marked @pytest.mark.slow
- Always run tests with: pytest -m "not slow"

## IDE
VS Code with Claude Code extension.

## Tech Stack
- Python 3.11+
- LLM: Groq API (llama-3.1-70b-versatile primary, llama-3.1-8b-instant fallback)
- Embeddings: BioLORD-2023-C via sentence-transformers (local after download)
- Vector Store: Qdrant (local/in-memory mode via qdrant-client)
- Knowledge Graph: NetworkX + JSON persistence (dev), Neo4j Community (prod)
- SQL Store: SQLite
- Agent Framework: LangGraph
- Backend: FastAPI
- Frontend: Streamlit
- Visualization: Pyvis (graph), Plotly (charts)

## Key Architecture Decisions
- **Graph store abstraction**: `GraphStore` base protocol with `NetworkXGraphStore` and `Neo4jGraphStore` implementations. Use NetworkX during dev (no server needed), Neo4j for demo/prod.
- **Multi-collection vector store**: Qdrant has separate collections per data source (faers_chunks, pubmed_chunks, trial_chunks, label_chunks). This enables source-specific retrieval.
- **Agentic loop via LangGraph**: The agent plans which sources to query, executes retrieval, evaluates sufficiency, and iterates (max 3 loops). This is NOT a simple retrieve-and-generate pipeline.
- **Evidence grading**: Every answer includes a grade (A-E) based on source authority and evidence convergence. Drug label confirmations outweigh individual FAERS reports.
- **FastAPI + Streamlit separation**: FastAPI is the real backend (stateless, testable, deployable). Streamlit calls the API. This shows production architecture thinking.

## Coding Standards
- Pydantic v2 models for ALL data schemas — no raw dicts flowing through the system
- Type hints on all functions
- Docstrings on all public functions and classes
- loguru for all logging (not print)
- All LLM prompts as module-level constants with version comments (# V1, # V2, etc.)
- Error handling: never crash the pipeline on a single bad record. Log and skip.
- Async FastAPI routes where beneficial

## Commands — What works NOW (dev machine)
```bash
pip install -r requirements.txt
python scripts/seed_sample_data.py               # generates sample JSON files (offline)
pytest tests/ -v -m "not slow"                    # local tests only
python -c "from src.config import settings; print(settings.PRIMARY_MODEL)"
uvicorn api.main:app --port 8000                  # API server (queries won't work)
streamlit run app/streamlit_app.py                # UI check (read-only without data)
```

## Commands — Run LATER on personal laptop
```bash
cp .env.example .env                              # add GROQ_API_KEY
python scripts/fetch_faers.py                     # download real FAERS data
python scripts/fetch_pubmed.py                    # download PubMed abstracts
python scripts/fetch_clinical_trials.py           # download trial data
python scripts/fetch_drug_labels.py               # download drug labels
python scripts/ingest_all.py                      # run all pipelines
python scripts/test_e2e.py                        # smoke test
pytest tests/ -v                                  # full suite
python eval/eval_retrieval.py                     # eval
python eval/eval_agent.py                         # agent eval
```

## Important Constants
- Qdrant collections: "faers_chunks", "pubmed_chunks", "trial_chunks", "label_chunks"
- Graph persistence: data/knowledge_graph.json (NetworkX mode)
- SQLite DB: data/medsignal.db
- LangGraph max iterations: 3
- Evidence grades: A (Strong), B (Moderate), C (Suggestive), D (Weak), E (Insufficient)
- All paths via src/config.py — never hardcoded

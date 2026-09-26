# Folder Structure

```
medsignal/
│
├── CLAUDE.md                         # Claude Code instructions
├── ARCHITECTURE.md                   # System architecture doc
├── FOLDER_STRUCTURE.md               # This file
├── README.md                         # Portfolio README
│
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variable template
├── .gitignore                        # Git ignore rules
├── pyproject.toml                    # Project metadata
├── .claude/settings.json             # Claude Code model config
│
├── data/
│   ├── sample/                       # Pre-built sample data (committed to git)
│   │   ├── faers/                    # ~50 sample FAERS reports as JSON
│   │   ├── pubmed/                   # ~30 sample PubMed abstracts as JSON
│   │   ├── clinical_trials/          # ~20 sample trial records as JSON
│   │   └── drug_labels/              # ~15 sample drug labels as JSON
│   ├── raw/                          # Downloaded real data (gitignored)
│   │   ├── faers/
│   │   ├── pubmed/
│   │   ├── clinical_trials/
│   │   └── drug_labels/
│   └── processed/                    # Post-extraction output (gitignored)
│       └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Central config via pydantic-settings
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py                # All Pydantic data models
│   │   └── enums.py                  # Enums (evidence grade, source type, etc.)
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── faers_loader.py           # Parse FAERS CSV/JSON → FaersReport
│   │   ├── pubmed_loader.py          # Parse PubMed XML/JSON → PubMedArticle
│   │   ├── clinical_trials_loader.py # Parse ClinicalTrials JSON → ClinicalTrial
│   │   ├── drug_label_loader.py      # Parse drug label JSON → DrugLabel
│   │   ├── chunker.py                # Source-aware chunking strategies
│   │   ├── extractor.py              # LLM extraction (drugs, AEs from text)
│   │   └── pipeline.py               # Orchestrates all 4 source pipelines
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── vector_store.py           # Qdrant wrapper (multi-collection)
│   │   ├── graph_store.py            # KG with Neo4j + NetworkX fallback
│   │   └── sql_store.py              # SQLite for metadata + structured data
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── query_planner.py          # Decompose query → sub-queries + sources
│   │   ├── faers_retriever.py        # FAERS-specific retrieval (vector + SQL)
│   │   ├── pubmed_retriever.py       # PubMed-specific retrieval
│   │   ├── trials_retriever.py       # Clinical trials retrieval
│   │   ├── label_retriever.py        # Drug label retrieval
│   │   ├── graph_retriever.py        # Knowledge graph traversal queries
│   │   ├── reranker.py               # Cross-source re-ranking by authority
│   │   └── evidence_grader.py        # Grade evidence strength (A-E)
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py                  # AgentState TypedDict
│   │   ├── nodes.py                  # LangGraph node functions
│   │   ├── edges.py                  # Conditional edge logic
│   │   └── graph.py                  # LangGraph graph definition + compilation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── llm_client.py             # Groq API wrapper with retry/fallback
│       ├── embedding.py              # BioLORD embedding helper
│       └── medical_terms.py          # Drug name normalization, MedDRA helpers
│
├── api/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app entry point
│   ├── dependencies.py               # Shared FastAPI dependencies (stores, agent)
│   └── routes/
│       ├── __init__.py
│       ├── query.py                  # POST /api/query — main RAG endpoint
│       ├── ingest.py                 # POST /api/ingest — trigger ingestion
│       ├── graph.py                  # GET /api/graph — graph exploration
│       ├── drugs.py                  # GET /api/drugs — drug lookup
│       └── analytics.py             # GET /api/analytics — stats + dashboards
│
├── app/
│   ├── streamlit_app.py              # Streamlit main entry
│   ├── pages/
│   │   ├── 1_query.py                # Natural language query interface
│   │   ├── 2_drug_explorer.py        # Browse drugs, see safety profile
│   │   ├── 3_signal_detection.py     # Emerging safety signal dashboard
│   │   ├── 4_graph_explorer.py       # Interactive knowledge graph
│   │   └── 5_analytics.py            # Data analytics + charts
│   └── components/
│       ├── __init__.py
│       ├── evidence_card.py          # Render graded evidence citations
│       ├── drug_profile.py           # Drug safety profile component
│       └── graph_viz.py              # Pyvis graph rendering
│
├── scripts/
│   ├── seed_sample_data.py           # Generate sample data (offline, no API)
│   ├── fetch_faers.py                # Download FAERS from openFDA API
│   ├── fetch_pubmed.py               # Download PubMed abstracts via E-utilities
│   ├── fetch_clinical_trials.py      # Download trials from ClinicalTrials.gov
│   ├── fetch_drug_labels.py          # Download labels from openFDA
│   ├── ingest_all.py                 # Run all ingestion pipelines
│   ├── reset_stores.py               # Clear stores
│   ├── query_cli.py                  # Interactive CLI query tool
│   └── test_e2e.py                   # End-to-end smoke test
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures
│   ├── test_faers_loader.py
│   ├── test_pubmed_loader.py
│   ├── test_chunker.py
│   ├── test_graph_store.py
│   ├── test_query_planner.py
│   ├── test_evidence_grader.py
│   ├── test_reranker.py
│   ├── test_agent.py
│   └── test_api.py
│
└── eval/
    ├── EVAL_LOG.md                   # Prompt version + accuracy tracking
    ├── eval_retrieval.py             # Evaluate retrieval quality
    ├── eval_agent.py                 # Evaluate agent planning + synthesis
    └── test_queries.json             # 20 graded test queries
```

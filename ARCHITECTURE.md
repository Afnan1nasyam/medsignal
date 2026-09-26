# MedSignal — Multi-Source Agentic RAG for Drug Safety Intelligence

## Vision
An agentic AI system that ingests drug safety data from four public sources (FDA FAERS, PubMed, ClinicalTrials.gov, DailyMed drug labels), builds a biomedical knowledge graph connecting drugs, adverse events, conditions, and publications, and uses LangGraph-based multi-step retrieval to answer complex pharmacovigilance questions with graded evidence citations.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                     │
│            Streamlit App (calls FastAPI backend)                      │
│  [Query] [Drug Explorer] [Signal Detection] [Graph] [Analytics]      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ HTTP
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        FastAPI BACKEND                                │
│   /api/query   /api/ingest   /api/graph   /api/analytics             │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     AGENTIC RAG ENGINE (LangGraph)                   │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐      │
│  │ Query       │───▶│ Execution    │───▶│ Evidence           │      │
│  │ Planner     │    │ Loop         │    │ Synthesizer        │      │
│  │             │    │              │    │                    │      │
│  │ - decompose │    │ - search     │    │ - merge findings   │      │
│  │ - pick src  │    │ - evaluate   │    │ - grade evidence   │      │
│  │ - plan      │    │ - iterate    │    │ - cite sources     │      │
│  └─────────────┘    └──────────────┘    └────────────────────┘      │
│         │                   │                      │                 │
│         ▼                   ▼                      ▼                 │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │              RETRIEVAL LAYER                                │     │
│  │  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌─────────────┐ │     │
│  │  │ FAERS    │ │ PubMed    │ │ Trials   │ │ Drug Labels │ │     │
│  │  │ Retriever│ │ Retriever │ │ Retriever│ │ Retriever   │ │     │
│  │  └──────────┘ └───────────┘ └──────────┘ └─────────────┘ │     │
│  │         │            │            │              │         │     │
│  │         └──────┬─────┴────────────┴──────────────┘         │     │
│  │                ▼                                           │     │
│  │         ┌─────────────┐                                    │     │
│  │         │  Re-ranker   │                                    │     │
│  │         └─────────────┘                                    │     │
│  └────────────────────────────────────────────────────────────┘     │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
              ┌─────────────────┼────────────────────┐
              ▼                 ▼                     ▼
┌───────────────────┐ ┌─────────────────┐ ┌──────────────────┐
│   Vector Store    │ │  Knowledge      │ │  SQL Store       │
│   (Qdrant)        │ │  Graph          │ │  (SQLite)        │
│                   │ │  (Neo4j /       │ │                  │
│ Collections:      │ │   NetworkX)     │ │ - drug metadata  │
│ - faers_chunks    │ │                 │ │ - FAERS reports  │
│ - pubmed_chunks   │ │ - drugs         │ │ - trial metadata │
│ - trial_chunks    │ │ - adverse evts  │ │ - query logs     │
│ - label_chunks    │ │ - conditions    │ │ - analytics      │
│                   │ │ - trials        │ │                  │
│                   │ │ - publications  │ │                  │
└───────────────────┘ └─────────────────┘ └──────────────────┘
              ▲                 ▲                     ▲
              └─────────────────┼────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINES                                │
│                                                                      │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ FAERS    │  │ PubMed    │  │ Clinical     │  │ Drug Label   │   │
│  │ Pipeline │  │ Pipeline  │  │ Trials       │  │ Pipeline     │   │
│  │          │  │           │  │ Pipeline     │  │              │   │
│  │ CSV/JSON │  │ XML/JSON  │  │ JSON API     │  │ XML/PDF      │   │
│  │ → parse  │  │ → parse   │  │ → parse      │  │ → parse      │   │
│  │ → extract│  │ → extract │  │ → extract    │  │ → extract    │   │
│  │ → embed  │  │ → embed   │  │ → embed      │  │ → embed      │   │
│  │ → store  │  │ → store   │  │ → store      │  │ → store      │   │
│  └──────────┘  └───────────┘  └──────────────┘  └──────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### 1. FDA FAERS (Adverse Event Reports)
- **Source**: openFDA API or quarterly data extracts (CSV)
- **URL**: https://open.fda.gov/data/faers/ and https://api.fda.gov/drug/event.json
- **Fields**: patient_age, patient_sex, drugs (name, indication, role), reactions (MedDRA preferred terms), outcomes (hospitalization, death, etc.), report_date, reporter_type
- **Volume**: Sample = 5,000 reports; Full = configurable via API pagination

### 2. PubMed Abstracts
- **Source**: NCBI E-utilities API (efetch, esearch)
- **URL**: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
- **Fields**: pmid, title, abstract, authors, journal, pub_date, mesh_terms, keywords
- **Volume**: Sample = 500 abstracts on drug safety topics

### 3. ClinicalTrials.gov
- **Source**: ClinicalTrials.gov REST API v2
- **URL**: https://clinicaltrials.gov/api/v2/studies
- **Fields**: nct_id, title, phase, status, conditions, interventions, adverse_events, eligibility, enrollment, start_date
- **Volume**: Sample = 300 trials

### 4. DailyMed / Drug Labels
- **Source**: DailyMed API or openFDA drug labeling endpoint
- **URL**: https://api.fda.gov/drug/label.json
- **Fields**: brand_name, generic_name, active_ingredient, warnings, adverse_reactions, contraindications, drug_interactions, indications_and_usage
- **Volume**: Sample = 200 drug labels

---

## Knowledge Graph Schema

### Nodes
| Node Type         | Key Properties                                                       |
|-------------------|----------------------------------------------------------------------|
| Drug              | name, generic_name, brand_names, drug_class, atc_code, manufacturer |
| AdverseEvent      | preferred_term, meddra_code, system_organ_class                      |
| Condition         | name, mesh_term, icd_code                                           |
| ClinicalTrial     | nct_id, title, phase, status, enrollment, start_date                |
| Publication       | pmid, title, journal, pub_date, pub_type                            |
| DrugClass         | name, atc_level, description                                        |

### Edges
| Edge Type             | From           | To             | Properties                                       |
|----------------------|----------------|----------------|--------------------------------------------------|
| REPORTED_WITH        | Drug           | AdverseEvent   | report_count, source (faers/trial/literature), seriousness_ratio |
| TREATS               | Drug           | Condition      | approval_status, source                          |
| CONTRAINDICATED_FOR  | Drug           | Condition      | source (label)                                   |
| INTERACTS_WITH       | Drug           | Drug           | severity, description, source                    |
| SAME_CLASS_AS        | Drug           | Drug           | via_atc_code                                     |
| BELONGS_TO           | Drug           | DrugClass      | —                                                |
| STUDIED_IN           | Drug           | ClinicalTrial  | role (intervention/comparator)                   |
| TRIAL_REPORTED       | ClinicalTrial  | AdverseEvent   | frequency, arm, severity                         |
| DESCRIBES            | Publication    | AdverseEvent   | —                                                |
| STUDIES_DRUG         | Publication    | Drug           | —                                                |
| AE_BELONGS_TO        | AdverseEvent   | SystemOrganClass | —                                              |

---

## Agent State Machine (LangGraph)

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         │
                         ▼
                ┌────────────────┐
                │ Query Planner  │
                │                │
                │ - classify     │
                │ - decompose    │
                │ - pick sources │
                └────────┬───────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Source Router         │
              │                      │
              │ Dispatches to 1-4    │
              │ source retrievers    │
              │ based on plan        │
              └──────────┬───────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
    ┌────────────┐ ┌──────────┐ ┌────────────┐
    │ FAERS      │ │ PubMed   │ │ Trials /   │
    │ Retriever  │ │ Retriever│ │ Labels     │
    └─────┬──────┘ └────┬─────┘ └─────┬──────┘
          │              │             │
          └──────────────┼─────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Evidence Evaluator   │
              │                      │
              │ - sufficient?        │
              │ - contradictions?    │
              │ - need more sources? │
              └──────────┬───────────┘
                         │
                    ┌────┴────┐
                    │         │
               Sufficient  Insufficient
                    │         │
                    │         └──▶ Back to Source Router
                    ▼              (max 3 iterations)
              ┌──────────────────────┐
              │ Evidence Synthesizer │
              │                      │
              │ - merge all findings │
              │ - grade evidence     │
              │ - generate answer    │
              │ - attach citations   │
              └──────────┬───────────┘
                         │
                         ▼
                    ┌──────────┐
                    │   END    │
                    └──────────┘
```

### Agent State Schema
```python
class AgentState(TypedDict):
    query: str                          # Original user query
    plan: QueryPlan                     # Decomposed sub-queries + source selections
    current_step: int                   # Which sub-query we're on
    max_iterations: int                 # Cap at 3 retrieval loops
    iteration: int                      # Current iteration
    retrieved_evidence: list[Evidence]  # All evidence gathered so far
    source_results: dict                # Results keyed by source name
    is_sufficient: bool                 # Does the evaluator think we have enough?
    contradictions: list[str]           # Any conflicting evidence found
    final_answer: str                   # Synthesized answer
    citations: list[Citation]           # Graded source citations
    evidence_grade: str                 # Overall evidence strength
    error: str | None                   # Any error during processing
```

---

## Evidence Grading System

| Grade | Label        | Criteria                                                              |
|-------|-------------|-----------------------------------------------------------------------|
| A     | Strong      | Confirmed in drug label + supported by clinical trial data            |
| B     | Moderate    | Multiple FAERS reports (>50) + supporting literature                  |
| C     | Suggestive  | Literature case reports + some FAERS signal                           |
| D     | Weak        | Isolated FAERS reports only, or single case report                    |
| E     | Insufficient| No evidence found in any source                                      |

Source authority ranking: Drug Label > Clinical Trial > Systematic Review > Case Report > FAERS Individual Report

---

## Tech Stack

| Component           | Technology                         | Why                                              |
|--------------------|------------------------------------|--------------------------------------------------|
| LLM                | Groq (llama-3.1-70b-versatile)    | Larger model for complex reasoning, free tier     |
| Embeddings         | BioLORD-2023-C (or PubMedBERT)   | Biomedical domain embeddings, much better than generic |
| Vector Store       | Qdrant (local mode)               | Better filtering + payloads than ChromaDB        |
| Knowledge Graph    | Neo4j Community (prod) / NetworkX (dev) | Real graph DB for complex traversals       |
| SQL Store          | SQLite                             | Metadata, FAERS structured data, query logs      |
| Agent Framework    | LangGraph                          | State machine for agentic retrieval loops        |
| Backend API        | FastAPI                            | Proper API layer, OpenAPI docs, async support    |
| Frontend           | Streamlit                          | Demo UI calling FastAPI                          |
| Data Pipeline      | Custom Python scripts              | Source-specific fetchers + parsers               |

---

## Extraction Schemas

### FAERS Report Extraction
```json
{
  "report_id": "string",
  "patient": {
    "age": "number | null",
    "age_unit": "year | month | day | null",
    "sex": "male | female | unknown",
    "weight": "number | null"
  },
  "drugs": [
    {
      "name": "string",
      "generic_name": "string | null",
      "role": "primary_suspect | secondary_suspect | concomitant | interacting",
      "indication": "string | null",
      "dosage": "string | null",
      "route": "string | null"
    }
  ],
  "reactions": [
    {
      "preferred_term": "string (MedDRA)",
      "outcome": "hospitalization | life_threatening | death | disability | congenital_anomaly | other"
    }
  ],
  "report_date": "YYYY-MM-DD",
  "reporter_type": "physician | pharmacist | consumer | other",
  "serious": true
}
```

### PubMed Abstract Extraction
```json
{
  "pmid": "string",
  "title": "string",
  "abstract": "string",
  "authors": ["string"],
  "journal": "string",
  "pub_date": "YYYY-MM-DD",
  "mesh_terms": ["string"],
  "drugs_mentioned": ["string"],
  "adverse_events_mentioned": ["string"],
  "study_type": "case_report | clinical_trial | meta_analysis | review | other",
  "key_findings": "string (LLM-extracted summary of findings)"
}
```

### Clinical Trial Extraction
```json
{
  "nct_id": "string",
  "title": "string",
  "phase": "1 | 2 | 3 | 4 | NA",
  "status": "completed | recruiting | terminated | withdrawn | other",
  "conditions": ["string"],
  "interventions": [
    {
      "name": "string",
      "type": "drug | biological | other"
    }
  ],
  "adverse_events": [
    {
      "term": "string",
      "organ_system": "string",
      "affected_count": "number",
      "at_risk_count": "number",
      "frequency_percent": "number"
    }
  ],
  "enrollment": "number",
  "start_date": "YYYY-MM-DD",
  "completion_date": "YYYY-MM-DD | null"
}
```

### Drug Label Extraction
```json
{
  "drug_name": "string",
  "generic_name": "string",
  "active_ingredient": "string",
  "manufacturer": "string",
  "indications": ["string"],
  "contraindications": ["string"],
  "warnings": ["string"],
  "adverse_reactions": [
    {
      "reaction": "string",
      "frequency": "common | uncommon | rare | very_rare | unknown",
      "description": "string"
    }
  ],
  "drug_interactions": [
    {
      "interacting_drug": "string",
      "severity": "major | moderate | minor",
      "description": "string"
    }
  ],
  "boxed_warning": "string | null"
}
```

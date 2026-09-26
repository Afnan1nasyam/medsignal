"""Central configuration for MedSignal.

All runtime settings are loaded here via ``pydantic-settings`` from environment
variables (and an optional ``.env`` file), with defaults mirroring
``.env.example``. Every path and tunable used elsewhere in the codebase must be
resolved through the module-level ``settings`` singleton — nothing is hardcoded
at the call site.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the environment / ``.env``.

    Field names match the environment variable names in ``.env.example``
    (matching is case-insensitive). Unknown environment variables are ignored
    so that unrelated shell state never breaks startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM & embeddings ---
    GROQ_API_KEY: str = ""
    PRIMARY_MODEL: str = "llama-3.1-70b-versatile"
    FALLBACK_MODEL: str = "llama-3.1-8b-instant"
    EMBEDDING_MODEL: str = "BioLORD-2023-C"

    # --- Knowledge graph backend ---
    GRAPH_BACKEND: Literal["networkx", "neo4j"] = "networkx"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "your_neo4j_password"

    # --- Paths ---
    DATA_DIR: str = "data"
    QDRANT_PATH: str = "data/qdrant"
    GRAPH_PATH: str = "data/knowledge_graph.json"
    SQLITE_PATH: str = "data/medsignal.db"

    # --- API settings ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # --- Rate limiting ---
    GROQ_RPM: int = 30
    GROQ_RETRY_ATTEMPTS: int = 3

    # --- Retrieval settings ---
    VECTOR_TOP_K: int = 10
    GRAPH_MAX_HOPS: int = 3
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # --- Agent settings ---
    AGENT_MAX_ITERATIONS: int = 3
    AGENT_MIN_EVIDENCE_PIECES: int = 3

    # --- External data-fetch APIs (used only by fetch scripts on the laptop) ---
    OPENFDA_API_KEY: str = ""
    NCBI_API_KEY: str = ""

    # --- Derived path helpers -------------------------------------------------
    @property
    def data_path(self) -> Path:
        """Root data directory as a :class:`~pathlib.Path`."""
        return Path(self.DATA_DIR)

    @property
    def sample_dir(self) -> Path:
        """Directory holding committed sample data."""
        return self.data_path / "sample"

    @property
    def sample_faers_dir(self) -> Path:
        """Directory of sample FAERS reports."""
        return self.sample_dir / "faers"

    @property
    def sample_pubmed_dir(self) -> Path:
        """Directory of sample PubMed abstracts."""
        return self.sample_dir / "pubmed"

    @property
    def sample_trials_dir(self) -> Path:
        """Directory of sample ClinicalTrials.gov records."""
        return self.sample_dir / "clinical_trials"

    @property
    def sample_labels_dir(self) -> Path:
        """Directory of sample drug labels."""
        return self.sample_dir / "drug_labels"


# Module-level singleton — import this everywhere instead of constructing Settings().
settings = Settings()

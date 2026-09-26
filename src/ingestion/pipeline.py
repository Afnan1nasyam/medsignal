"""End-to-end ingestion orchestrator.

Ties the per-source loaders, the source-aware chunker, the embedding model, and
the three stores (SQL, knowledge graph, Qdrant vectors) into one pipeline. Each
``ingest_*`` method loads validated records, persists them to SQL, builds the
knowledge-graph nodes/edges, and chunks + embeds + stores the text for retrieval.

Design notes:

* **Resilient by record.** Graph construction for a single record is wrapped in
  ``try/except`` so one malformed record never aborts the batch (loaders already
  skip unparseable records).
* **Offline-safe.** Constructing the pipeline performs no network I/O: the
  embedding model is lazy-loaded on first embed, and the LLM extractor is only
  built when ``enrich=True`` is requested. The embed + vector-store step is
  wrapped defensively so an unavailable embedding model (e.g. a blocked model
  download) degrades to "SQL + graph only" rather than crashing.
* **Meaningful graph weights.** FAERS ``REPORTED_WITH`` edges are aggregated so
  ``report_count`` reflects how many reports link a drug to a reaction.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from loguru import logger

from src.config import settings
from src.ingestion.chunker import (
    Chunk,
    chunk_clinical_trial,
    chunk_drug_label,
    chunk_faers_report,
    chunk_pubmed_article,
)
from src.ingestion.clinical_trials_loader import load_clinical_trials
from src.ingestion.drug_label_loader import load_drug_labels
from src.ingestion.faers_loader import load_faers_reports
from src.ingestion.pubmed_loader import load_pubmed_articles
from src.models.enums import DataSource, EdgeType
from src.storage.graph_store import get_graph_store
from src.storage.sql_store import SQLStore
from src.storage.vector_store import QdrantVectorStore
from src.utils.embedding import get_embedding_model
from src.utils.medical_terms import DRUG_CLASSES, normalize_drug_name

# Reverse lookup: generic drug name -> its drug class (for SAME_CLASS_AS edges).
_DRUG_TO_CLASS: dict[str, str] = {
    drug: cls for cls, drugs in DRUG_CLASSES.items() for drug in drugs
}


class IngestionPipeline:
    """Loads, stores, graphs, chunks and embeds all four data sources."""

    def __init__(self, use_extractor: bool = False) -> None:
        """Initialize every store and the embedding model.

        Args:
            use_extractor: If True, eagerly construct a :class:`MedicalExtractor`
                (needs the Groq API). Left False by default so the pipeline is
                offline-safe; an extractor is otherwise built lazily only when
                ``ingest_pubmed(enrich=True)`` is called.
        """
        self.vector_store = QdrantVectorStore()
        self.graph = get_graph_store()
        self.sql = SQLStore()
        self.embedding = get_embedding_model()
        self.extractor = None
        if use_extractor:
            self.extractor = self._build_extractor()

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _build_extractor():
        """Construct a :class:`MedicalExtractor`, or ``None`` if unavailable."""
        try:
            from src.ingestion.extractor import MedicalExtractor

            return MedicalExtractor()
        except Exception as exc:  # noqa: BLE001 - Groq/deps may be unavailable
            logger.warning("MedicalExtractor unavailable ({}); enrichment disabled", exc)
            return None

    def _get_extractor(self):
        """Return a cached extractor, lazily building one on first request."""
        if self.extractor is None:
            self.extractor = self._build_extractor()
        return self.extractor

    def _add_drug_node(self, generic: str, extra: dict | None = None) -> str:
        """Upsert a drug node keyed by its normalized generic name; return its id."""
        props: dict = {"generic_name": generic}
        drug_class = _DRUG_TO_CLASS.get(generic)
        if drug_class:
            props["drug_class"] = drug_class
        if extra:
            props.update({k: v for k, v in extra.items() if v is not None})
        return self.graph.add_drug(generic, props)

    def _embed_and_store(self, source: DataSource, chunks: list[Chunk]) -> None:
        """Embed chunk texts and upsert them into the source's Qdrant collection.

        Any failure to load the embedding model (e.g. a blocked download) is
        logged and swallowed so SQL + graph ingestion still succeeds.
        """
        if not chunks:
            logger.info("No chunks to embed for {}", source.value)
            return
        texts = [text for text, _ in chunks]
        metadatas = [meta.model_dump(mode="json") for _, meta in chunks]
        try:
            embeddings = self.embedding.embed_batch(texts)
            self.vector_store.add_chunks(
                source=source, texts=texts, metadatas=metadatas, embeddings=embeddings
            )
            logger.info("Embedded + stored {} {} chunks", len(texts), source.value)
        except Exception as exc:  # noqa: BLE001 - embedding is best-effort here
            logger.error(
                "Embedding/vector storage skipped for {} ({} chunks): {}",
                source.value,
                len(texts),
                exc,
            )

    # -- FAERS ------------------------------------------------------------- #
    def ingest_faers(self, data_dir: Path) -> int:
        """Ingest FAERS reports into SQL, the graph, and the vector store.

        Builds ``Drug -REPORTED_WITH-> AdverseEvent`` edges (with an aggregated
        ``report_count``) and ``Drug -TREATS-> Condition`` edges from indications.

        Args:
            data_dir: Directory of FAERS JSON files.

        Returns:
            The number of reports ingested.
        """
        reports = load_faers_reports(Path(data_dir))
        if not reports:
            return 0
        self.sql.save_faers_batch(reports)

        drug_ids: dict[str, str] = {}
        ae_ids: dict[str, str] = {}
        cond_ids: dict[str, str] = {}
        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_outcome: dict[tuple[str, str], str] = {}
        treats: set[tuple[str, str]] = set()
        all_chunks: list[Chunk] = []

        for report in reports:
            try:
                report_generics: list[str] = []
                for drug in report.drugs:
                    generic = (drug.generic_name or normalize_drug_name(drug.name)).strip().lower()
                    if not generic:
                        continue
                    report_generics.append(generic)
                    if generic not in drug_ids:
                        drug_ids[generic] = self._add_drug_node(generic)
                    if drug.indication:
                        treats.add((generic, drug.indication.strip()))
                for reaction in report.reactions:
                    term = reaction.preferred_term.strip()
                    if not term:
                        continue
                    if term not in ae_ids:
                        ae_ids[term] = self.graph.add_adverse_event(term, {})
                    for generic in report_generics:
                        pair_counts[(generic, term)] += 1
                        pair_outcome.setdefault((generic, term), reaction.outcome.value)
            except Exception as exc:  # noqa: BLE001 - never abort the batch
                logger.warning("FAERS graph build failed for {}: {}", report.report_id, exc)
            all_chunks.extend(chunk_faers_report(report))

        for (generic, term), count in pair_counts.items():
            self.graph.add_edge(
                drug_ids[generic],
                ae_ids[term],
                EdgeType.REPORTED_WITH,
                {"report_count": count, "source": "FAERS", "outcome": pair_outcome[(generic, term)]},
            )
        for generic, indication in treats:
            if indication not in cond_ids:
                cond_ids[indication] = self.graph.add_condition(indication, {})
            self.graph.add_edge(
                drug_ids[generic], cond_ids[indication], EdgeType.TREATS, {"source": "FAERS"}
            )

        self._embed_and_store(DataSource.FAERS, all_chunks)
        logger.info("Ingested {} FAERS reports", len(reports))
        return len(reports)

    # -- PubMed ------------------------------------------------------------ #
    def ingest_pubmed(self, data_dir: Path, enrich: bool = False) -> int:
        """Ingest PubMed articles into SQL, the graph, and the vector store.

        Builds ``Publication -STUDIES_DRUG-> Drug`` and
        ``Publication -DESCRIBES-> AdverseEvent`` edges. When ``enrich`` is True
        and an extractor is available, articles are LLM-enriched first; if the
        extractor cannot be built (offline), enrichment is skipped.

        Args:
            data_dir: Directory of PubMed JSON files.
            enrich: Whether to run LLM entity extraction before ingestion.

        Returns:
            The number of articles ingested.
        """
        articles = load_pubmed_articles(Path(data_dir))
        if not articles:
            return 0

        if enrich:
            extractor = self._get_extractor()
            if extractor is not None:
                try:
                    articles = extractor.enrich_batch(articles)
                except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
                    logger.warning("PubMed enrichment failed; using raw articles: {}", exc)

        self.sql.save_pubmed_batch(articles)

        all_chunks: list[Chunk] = []
        for article in articles:
            try:
                pub_id = self.graph.add_publication(
                    article.pmid,
                    {"title": article.title, "journal": article.journal, "pub_date": article.pub_date},
                )
                for drug in article.drugs_mentioned:
                    generic = normalize_drug_name(drug)
                    if not generic:
                        continue
                    drug_id = self._add_drug_node(generic)
                    self.graph.add_edge(pub_id, drug_id, EdgeType.STUDIES_DRUG, {})
                for ae in article.adverse_events_mentioned:
                    term = str(ae).strip()
                    if not term:
                        continue
                    ae_id = self.graph.add_adverse_event(term, {})
                    self.graph.add_edge(pub_id, ae_id, EdgeType.DESCRIBES, {})
            except Exception as exc:  # noqa: BLE001 - never abort the batch
                logger.warning("PubMed graph build failed for PMID {}: {}", article.pmid, exc)
            all_chunks.extend(chunk_pubmed_article(article))

        self._embed_and_store(DataSource.PUBMED, all_chunks)
        logger.info("Ingested {} PubMed articles", len(articles))
        return len(articles)

    # -- Clinical trials --------------------------------------------------- #
    def ingest_clinical_trials(self, data_dir: Path) -> int:
        """Ingest clinical trials into SQL, the graph, and the vector store.

        Builds ``Drug -STUDIED_IN-> ClinicalTrial`` and
        ``ClinicalTrial -TRIAL_REPORTED-> AdverseEvent`` edges.

        Args:
            data_dir: Directory of clinical-trial JSON files.

        Returns:
            The number of trials ingested.
        """
        trials = load_clinical_trials(Path(data_dir))
        if not trials:
            return 0
        self.sql.save_trial_batch(trials)

        all_chunks: list[Chunk] = []
        for trial in trials:
            try:
                trial_id = self.graph.add_clinical_trial(
                    trial.nct_id,
                    {"title": trial.title, "phase": trial.phase.value, "status": trial.status.value},
                )
                for intervention in trial.interventions:
                    generic = normalize_drug_name(intervention.name)
                    if not generic:
                        continue
                    drug_id = self._add_drug_node(generic)
                    self.graph.add_edge(drug_id, trial_id, EdgeType.STUDIED_IN, {})
                for ae in trial.adverse_events:
                    term = ae.term.strip()
                    if not term:
                        continue
                    ae_id = self.graph.add_adverse_event(term, {})
                    self.graph.add_edge(
                        trial_id,
                        ae_id,
                        EdgeType.TRIAL_REPORTED,
                        {
                            "frequency_percent": ae.frequency_percent,
                            "affected_count": ae.affected_count,
                            "at_risk_count": ae.at_risk_count,
                        },
                    )
            except Exception as exc:  # noqa: BLE001 - never abort the batch
                logger.warning("Trial graph build failed for {}: {}", trial.nct_id, exc)
            all_chunks.extend(chunk_clinical_trial(trial))

        self._embed_and_store(DataSource.CLINICAL_TRIALS, all_chunks)
        logger.info("Ingested {} clinical trials", len(trials))
        return len(trials)

    # -- Drug labels ------------------------------------------------------- #
    def ingest_drug_labels(self, data_dir: Path) -> int:
        """Ingest drug labels into SQL, the graph, and the vector store.

        Builds ``Drug -TREATS-> Condition`` (indications),
        ``Drug -CONTRAINDICATED_FOR-> Condition``, ``Drug -INTERACTS_WITH-> Drug``,
        and ``Drug -REPORTED_WITH-> AdverseEvent`` (label reactions) edges, then
        adds ``SAME_CLASS_AS`` / ``BELONGS_TO`` edges from ``DRUG_CLASSES``.

        Args:
            data_dir: Directory of drug-label JSON files.

        Returns:
            The number of labels ingested.
        """
        labels = load_drug_labels(Path(data_dir))
        if not labels:
            return 0
        self.sql.save_label_batch(labels)

        label_generics: set[str] = set()
        cond_ids: dict[str, str] = {}
        all_chunks: list[Chunk] = []

        for label in labels:
            try:
                generic = normalize_drug_name(label.generic_name or label.drug_name)
                if not generic:
                    continue
                drug_id = self._add_drug_node(
                    generic,
                    {
                        "brand": label.drug_name,
                        "active_ingredient": label.active_ingredient,
                        "manufacturer": label.manufacturer,
                    },
                )
                label_generics.add(generic)

                for indication in label.indications:
                    cond = indication.strip()
                    if not cond:
                        continue
                    if cond not in cond_ids:
                        cond_ids[cond] = self.graph.add_condition(cond, {})
                    self.graph.add_edge(drug_id, cond_ids[cond], EdgeType.TREATS, {"source": "label"})
                for contra in label.contraindications:
                    cond = contra.strip()
                    if not cond:
                        continue
                    if cond not in cond_ids:
                        cond_ids[cond] = self.graph.add_condition(cond, {})
                    self.graph.add_edge(drug_id, cond_ids[cond], EdgeType.CONTRAINDICATED_FOR, {})
                for interaction in label.drug_interactions:
                    other_generic = normalize_drug_name(interaction.interacting_drug)
                    if not other_generic or other_generic == generic:
                        continue
                    other_id = self._add_drug_node(other_generic)
                    self.graph.add_edge(
                        drug_id,
                        other_id,
                        EdgeType.INTERACTS_WITH,
                        {"severity": interaction.severity.value, "description": interaction.description},
                    )
                for reaction in label.adverse_reactions:
                    term = reaction.reaction.strip()
                    if not term:
                        continue
                    ae_id = self.graph.add_adverse_event(term, {})
                    self.graph.add_edge(
                        drug_id,
                        ae_id,
                        EdgeType.REPORTED_WITH,
                        {"frequency": reaction.frequency.value, "source": "label"},
                    )
            except Exception as exc:  # noqa: BLE001 - never abort the batch
                logger.warning("Label graph build failed for {}: {}", label.drug_name, exc)
            all_chunks.extend(chunk_drug_label(label))

        self._build_class_edges(label_generics)
        self._embed_and_store(DataSource.DRUG_LABELS, all_chunks)
        logger.info("Ingested {} drug labels", len(labels))
        return len(labels)

    def _build_class_edges(self, generics: set[str]) -> None:
        """Connect same-class drugs via ``BELONGS_TO`` and ``SAME_CLASS_AS``.

        Only drugs actually present in ``generics`` are linked, so classes with
        a single ingested member produce a class node but no sibling edges.
        """
        for drug_class, members in DRUG_CLASSES.items():
            present = [m for m in members if m in generics]
            if not present:
                continue
            class_id = self.graph.add_drug_class(drug_class, {})
            member_ids = {
                m: self._add_drug_node(m) for m in present
            }
            for member, node_id in member_ids.items():
                self.graph.add_edge(node_id, class_id, EdgeType.BELONGS_TO, {})
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    self.graph.add_edge(
                        member_ids[present[i]],
                        member_ids[present[j]],
                        EdgeType.SAME_CLASS_AS,
                        {"drug_class": drug_class},
                    )

    # -- Orchestration ----------------------------------------------------- #
    def ingest_all(self, data_dir: Path | None = None, enrich: bool = False) -> dict:
        """Run all four pipelines in order, persist the graph, and summarize.

        Args:
            data_dir: Base directory holding the four source subdirectories
                (``faers``, ``pubmed``, ``clinical_trials``, ``drug_labels``).
                Defaults to ``settings.DATA_DIR / "sample"``.
            enrich: Whether to LLM-enrich PubMed articles.

        Returns:
            A summary dict with per-source counts, the total, elapsed time, and
            graph + store statistics.
        """
        base = Path(data_dir) if data_dir else Path(settings.DATA_DIR) / "sample"
        logger.info("Starting full ingestion from {}", base)
        start = time.perf_counter()

        counts = {
            "faers": self.ingest_faers(base / "faers"),
            "pubmed": self.ingest_pubmed(base / "pubmed", enrich=enrich),
            "trials": self.ingest_clinical_trials(base / "clinical_trials"),
            "labels": self.ingest_drug_labels(base / "drug_labels"),
        }

        # Persist the NetworkX graph (Neo4j persists server-side, no save()).
        if hasattr(self.graph, "save"):
            self.graph.save()

        elapsed = round(time.perf_counter() - start, 2)
        summary = {
            "counts": counts,
            "total_records": sum(counts.values()),
            "total_time_seconds": elapsed,
            "graph_stats": self.graph.get_stats(),
            "store_stats": self.get_stats(),
        }
        logger.info("Ingestion complete in {}s: {}", elapsed, counts)
        return summary

    def reset_stores(self) -> None:
        """Clear all three stores (vector, graph, SQL) before a fresh ingest."""
        self.vector_store.reset()
        self.graph.reset()
        self.sql.reset()
        logger.info("All stores reset")

    def get_stats(self) -> dict:
        """Return combined statistics from the SQL, vector, and graph stores."""
        return {
            "sql": self.sql.get_all_stats(),
            "vector": self.vector_store.get_stats(),
            "graph": self.graph.get_stats(),
        }

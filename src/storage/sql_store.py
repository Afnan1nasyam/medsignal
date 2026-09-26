"""SQLite store for structured source records and query logging.

Holds the raw, structured payloads for all four data sources (FAERS, PubMed,
ClinicalTrials.gov, drug labels) alongside a ``query_log`` audit table. Nested
Pydantic sub-objects and lists are serialized to JSON text columns; on read they
are parsed back into Python objects with the ``_json`` suffix stripped from the
key (e.g. the ``drugs_json`` column is returned as ``drugs``).

Runs fully offline against a local SQLite file at ``settings.SQLITE_PATH``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import settings
from src.models.schemas import (
    ClinicalTrialRecord,
    DrugLabel,
    FaersReport,
    PubMedArticle,
)

# --------------------------------------------------------------------------- #
# Table DDL
# --------------------------------------------------------------------------- #
_SCHEMA: dict[str, str] = {
    "faers_reports": """
        CREATE TABLE IF NOT EXISTS faers_reports (
            report_id      TEXT PRIMARY KEY,
            patient_json   TEXT,
            drugs_json     TEXT,
            reactions_json TEXT,
            report_date    TEXT,
            reporter_type  TEXT,
            serious        INTEGER,
            created_at     TEXT
        )
    """,
    "pubmed_articles": """
        CREATE TABLE IF NOT EXISTS pubmed_articles (
            pmid                 TEXT PRIMARY KEY,
            title                TEXT,
            abstract             TEXT,
            authors_json         TEXT,
            journal              TEXT,
            pub_date             TEXT,
            mesh_terms_json      TEXT,
            drugs_mentioned_json TEXT,
            adverse_events_json  TEXT,
            study_type           TEXT,
            key_findings         TEXT,
            created_at           TEXT
        )
    """,
    "clinical_trials": """
        CREATE TABLE IF NOT EXISTS clinical_trials (
            nct_id             TEXT PRIMARY KEY,
            title              TEXT,
            phase              TEXT,
            status             TEXT,
            conditions_json    TEXT,
            interventions_json TEXT,
            adverse_events_json TEXT,
            enrollment         INTEGER,
            start_date         TEXT,
            completion_date    TEXT,
            created_at         TEXT
        )
    """,
    "drug_labels": """
        CREATE TABLE IF NOT EXISTS drug_labels (
            drug_name              TEXT PRIMARY KEY,
            generic_name           TEXT,
            active_ingredient      TEXT,
            manufacturer           TEXT,
            indications_json       TEXT,
            contraindications_json TEXT,
            warnings_json          TEXT,
            adverse_reactions_json TEXT,
            drug_interactions_json TEXT,
            boxed_warning          TEXT,
            created_at             TEXT
        )
    """,
    "query_log": """
        CREATE TABLE IF NOT EXISTS query_log (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            query              TEXT,
            intent             TEXT,
            sources_consulted  TEXT,
            evidence_grade     TEXT,
            result_count       INTEGER,
            processing_time    REAL,
            timestamp          TEXT
        )
    """,
}


def _now() -> str:
    """Current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class SQLStore:
    """A thin SQLite wrapper over the structured source tables and query log."""

    def __init__(self) -> None:
        """Open the SQLite database and create all tables if they don't exist."""
        self.path = Path(settings.SQLITE_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ``check_same_thread=False`` lets FastAPI's threadpool share the store.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("SQLStore ready at {}", self.path)

    # -- schema management ------------------------------------------------- #
    def _create_tables(self) -> None:
        """Create every table if it is not already present."""
        for ddl in _SCHEMA.values():
            self.conn.execute(ddl)
        self.conn.commit()

    def reset(self) -> None:
        """Drop and recreate all tables (clears every row)."""
        for table in _SCHEMA:
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")
        self._create_tables()
        logger.info("SQLStore reset: dropped and recreated {} tables", len(_SCHEMA))

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    # -- row helpers ------------------------------------------------------- #
    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
        """Convert a row into a dict, parsing ``*_json`` columns back to Python.

        The ``_json`` suffix is stripped from the resulting key and the SQLite
        ``serious`` integer is coerced back to a bool.
        """
        if row is None:
            return None
        out: dict[str, Any] = {}
        for key in row.keys():
            value = row[key]
            if key.endswith("_json"):
                out[key[:-5]] = json.loads(value) if value else None
            elif key == "serious":
                out[key] = bool(value) if value is not None else None
            else:
                out[key] = value
        return out

    def _rows_to_dicts(self, rows: list[sqlite3.Row]) -> list[dict]:
        """Convert a list of rows to a list of parsed dicts."""
        return [self._row_to_dict(row) for row in rows]  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    # FAERS
    # ------------------------------------------------------------------ #
    @staticmethod
    def _faers_row(report: FaersReport) -> tuple:
        """Flatten a :class:`FaersReport` into an insert tuple."""
        data = report.model_dump(mode="json")
        return (
            data["report_id"],
            json.dumps(data["patient"]),
            json.dumps(data["drugs"]),
            json.dumps(data["reactions"]),
            data.get("report_date"),
            data["reporter_type"],
            int(bool(data["serious"])),
            _now(),
        )

    _FAERS_INSERT = (
        "INSERT OR REPLACE INTO faers_reports "
        "(report_id, patient_json, drugs_json, reactions_json, report_date, "
        "reporter_type, serious, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )

    def save_faers_report(self, report: FaersReport) -> None:
        """Upsert a single FAERS report (nested objects stored as JSON)."""
        self.conn.execute(self._FAERS_INSERT, self._faers_row(report))
        self.conn.commit()

    def save_faers_batch(self, reports: list[FaersReport]) -> None:
        """Batch-upsert FAERS reports with a single ``executemany``."""
        rows = [self._faers_row(r) for r in reports]
        self.conn.executemany(self._FAERS_INSERT, rows)
        self.conn.commit()
        logger.info("Saved {} FAERS reports", len(rows))

    def get_faers_report(self, report_id: str) -> dict | None:
        """Fetch one FAERS report by id, or ``None`` if absent."""
        cur = self.conn.execute(
            "SELECT * FROM faers_reports WHERE report_id = ?", (report_id,)
        )
        return self._row_to_dict(cur.fetchone())

    def search_faers(
        self,
        drug_name: str | None = None,
        reaction: str | None = None,
        serious: bool | None = None,
    ) -> list[dict]:
        """Search FAERS reports by drug, reaction and/or seriousness.

        Drug and reaction are matched case-insensitively against the JSON text
        of the ``drugs_json`` / ``reactions_json`` columns.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if drug_name:
            clauses.append("drugs_json LIKE ?")
            params.append(f"%{drug_name}%")
        if reaction:
            clauses.append("reactions_json LIKE ?")
            params.append(f"%{reaction}%")
        if serious is not None:
            clauses.append("serious = ?")
            params.append(int(serious))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = self.conn.execute(f"SELECT * FROM faers_reports{where}", params)
        return self._rows_to_dicts(cur.fetchall())

    def count_faers_by_drug_reaction(self, drug_name: str, reaction: str) -> int:
        """Count FAERS reports mentioning both a drug and a reaction."""
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM faers_reports "
            "WHERE drugs_json LIKE ? AND reactions_json LIKE ?",
            (f"%{drug_name}%", f"%{reaction}%"),
        )
        return int(cur.fetchone()[0])

    # ------------------------------------------------------------------ #
    # PubMed
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pubmed_row(article: PubMedArticle) -> tuple:
        """Flatten a :class:`PubMedArticle` into an insert tuple."""
        data = article.model_dump(mode="json")
        return (
            data["pmid"],
            data["title"],
            data["abstract"],
            json.dumps(data["authors"]),
            data["journal"],
            data.get("pub_date"),
            json.dumps(data["mesh_terms"]),
            json.dumps(data["drugs_mentioned"]),
            json.dumps(data["adverse_events_mentioned"]),
            data["study_type"],
            data["key_findings"],
            _now(),
        )

    _PUBMED_INSERT = (
        "INSERT OR REPLACE INTO pubmed_articles "
        "(pmid, title, abstract, authors_json, journal, pub_date, "
        "mesh_terms_json, drugs_mentioned_json, adverse_events_json, "
        "study_type, key_findings, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    def save_pubmed_article(self, article: PubMedArticle) -> None:
        """Upsert a single PubMed article."""
        self.conn.execute(self._PUBMED_INSERT, self._pubmed_row(article))
        self.conn.commit()

    def save_pubmed_batch(self, articles: list[PubMedArticle]) -> None:
        """Batch-upsert PubMed articles."""
        rows = [self._pubmed_row(a) for a in articles]
        self.conn.executemany(self._PUBMED_INSERT, rows)
        self.conn.commit()
        logger.info("Saved {} PubMed articles", len(rows))

    def get_pubmed_article(self, pmid: str) -> dict | None:
        """Fetch one PubMed article by PMID, or ``None`` if absent."""
        cur = self.conn.execute(
            "SELECT * FROM pubmed_articles WHERE pmid = ?", (pmid,)
        )
        return self._row_to_dict(cur.fetchone())

    def search_pubmed(
        self, drug: str | None = None, adverse_event: str | None = None
    ) -> list[dict]:
        """Search PubMed articles by mentioned drug and/or adverse event."""
        clauses: list[str] = []
        params: list[Any] = []
        if drug:
            clauses.append("drugs_mentioned_json LIKE ?")
            params.append(f"%{drug}%")
        if adverse_event:
            clauses.append("adverse_events_json LIKE ?")
            params.append(f"%{adverse_event}%")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = self.conn.execute(f"SELECT * FROM pubmed_articles{where}", params)
        return self._rows_to_dicts(cur.fetchall())

    # ------------------------------------------------------------------ #
    # Clinical trials
    # ------------------------------------------------------------------ #
    @staticmethod
    def _trial_row(trial: ClinicalTrialRecord) -> tuple:
        """Flatten a :class:`ClinicalTrialRecord` into an insert tuple."""
        data = trial.model_dump(mode="json")
        return (
            data["nct_id"],
            data["title"],
            data["phase"],
            data["status"],
            json.dumps(data["conditions"]),
            json.dumps(data["interventions"]),
            json.dumps(data["adverse_events"]),
            data.get("enrollment"),
            data.get("start_date"),
            data.get("completion_date"),
            _now(),
        )

    _TRIAL_INSERT = (
        "INSERT OR REPLACE INTO clinical_trials "
        "(nct_id, title, phase, status, conditions_json, interventions_json, "
        "adverse_events_json, enrollment, start_date, completion_date, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    def save_clinical_trial(self, trial: ClinicalTrialRecord) -> None:
        """Upsert a single clinical trial."""
        self.conn.execute(self._TRIAL_INSERT, self._trial_row(trial))
        self.conn.commit()

    def save_trial_batch(self, trials: list[ClinicalTrialRecord]) -> None:
        """Batch-upsert clinical trials."""
        rows = [self._trial_row(t) for t in trials]
        self.conn.executemany(self._TRIAL_INSERT, rows)
        self.conn.commit()
        logger.info("Saved {} clinical trials", len(rows))

    def get_clinical_trial(self, nct_id: str) -> dict | None:
        """Fetch one clinical trial by NCT id, or ``None`` if absent."""
        cur = self.conn.execute(
            "SELECT * FROM clinical_trials WHERE nct_id = ?", (nct_id,)
        )
        return self._row_to_dict(cur.fetchone())

    def search_trials(
        self,
        drug: str | None = None,
        condition: str | None = None,
        phase: str | None = None,
    ) -> list[dict]:
        """Search trials by intervention drug, condition and/or phase."""
        clauses: list[str] = []
        params: list[Any] = []
        if drug:
            clauses.append("interventions_json LIKE ?")
            params.append(f"%{drug}%")
        if condition:
            clauses.append("conditions_json LIKE ?")
            params.append(f"%{condition}%")
        if phase:
            clauses.append("phase = ?")
            params.append(phase)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = self.conn.execute(f"SELECT * FROM clinical_trials{where}", params)
        return self._rows_to_dicts(cur.fetchall())

    # ------------------------------------------------------------------ #
    # Drug labels
    # ------------------------------------------------------------------ #
    @staticmethod
    def _label_row(label: DrugLabel) -> tuple:
        """Flatten a :class:`DrugLabel` into an insert tuple."""
        data = label.model_dump(mode="json")
        return (
            data["drug_name"],
            data["generic_name"],
            data["active_ingredient"],
            data.get("manufacturer"),
            json.dumps(data["indications"]),
            json.dumps(data["contraindications"]),
            json.dumps(data["warnings"]),
            json.dumps(data["adverse_reactions"]),
            json.dumps(data["drug_interactions"]),
            data.get("boxed_warning"),
            _now(),
        )

    _LABEL_INSERT = (
        "INSERT OR REPLACE INTO drug_labels "
        "(drug_name, generic_name, active_ingredient, manufacturer, "
        "indications_json, contraindications_json, warnings_json, "
        "adverse_reactions_json, drug_interactions_json, boxed_warning, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    def save_drug_label(self, label: DrugLabel) -> None:
        """Upsert a single drug label."""
        self.conn.execute(self._LABEL_INSERT, self._label_row(label))
        self.conn.commit()

    def save_label_batch(self, labels: list[DrugLabel]) -> None:
        """Batch-upsert drug labels."""
        rows = [self._label_row(lb) for lb in labels]
        self.conn.executemany(self._LABEL_INSERT, rows)
        self.conn.commit()
        logger.info("Saved {} drug labels", len(rows))

    def get_drug_label(self, drug_name: str) -> dict | None:
        """Fetch one drug label by drug name, or ``None`` if absent."""
        cur = self.conn.execute(
            "SELECT * FROM drug_labels WHERE drug_name = ?", (drug_name,)
        )
        return self._row_to_dict(cur.fetchone())

    def search_labels(self, drug: str | None = None) -> list[dict]:
        """Search drug labels by brand or generic name (case-insensitive)."""
        if drug:
            cur = self.conn.execute(
                "SELECT * FROM drug_labels "
                "WHERE drug_name LIKE ? OR generic_name LIKE ?",
                (f"%{drug}%", f"%{drug}%"),
            )
        else:
            cur = self.conn.execute("SELECT * FROM drug_labels")
        return self._rows_to_dicts(cur.fetchall())

    # ------------------------------------------------------------------ #
    # Query log & stats
    # ------------------------------------------------------------------ #
    def log_query(
        self,
        query: str,
        intent: str | None,
        sources: Any,
        evidence_grade: str | None,
        result_count: int,
        processing_time: float,
    ) -> None:
        """Append a row to the query audit log.

        ``sources`` may be a list (stored as JSON) or a string (stored as-is).
        """
        if isinstance(sources, (list, tuple)):
            sources_text = json.dumps(list(sources))
        else:
            sources_text = str(sources) if sources is not None else None
        self.conn.execute(
            "INSERT INTO query_log "
            "(query, intent, sources_consulted, evidence_grade, result_count, "
            "processing_time, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                query,
                intent,
                sources_text,
                evidence_grade,
                result_count,
                processing_time,
                _now(),
            ),
        )
        self.conn.commit()

    def get_query_stats(self) -> dict:
        """Summarize the query log: totals, by-intent, by-grade, avg time."""
        total = self.conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        by_intent = {
            row["intent"]: row["n"]
            for row in self.conn.execute(
                "SELECT intent, COUNT(*) AS n FROM query_log GROUP BY intent"
            ).fetchall()
        }
        by_grade = {
            row["evidence_grade"]: row["n"]
            for row in self.conn.execute(
                "SELECT evidence_grade, COUNT(*) AS n FROM query_log "
                "GROUP BY evidence_grade"
            ).fetchall()
        }
        avg_time = self.conn.execute(
            "SELECT AVG(processing_time) FROM query_log"
        ).fetchone()[0]
        return {
            "total_queries": int(total),
            "by_intent": by_intent,
            "by_grade": by_grade,
            "avg_processing_time": float(avg_time) if avg_time is not None else 0.0,
        }

    def get_all_stats(self) -> dict:
        """Return a row count for each table."""
        stats: dict[str, int] = {}
        for table in _SCHEMA:
            count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[table] = int(count)
        return stats

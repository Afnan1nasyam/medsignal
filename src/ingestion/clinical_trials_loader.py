"""Loader for ClinicalTrials.gov studies (local JSON files only).

Parses both our simplified sample format (already shaped like
:class:`ClinicalTrialRecord`) and the ClinicalTrials.gov API v2 format
(``studies[].protocolSection`` / ``resultsSection``) into validated
:class:`ClinicalTrialRecord` models. Malformed records are logged and skipped.
"""

from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError

from src.ingestion._common import extract_items, read_json_documents
from src.models.enums import TrialPhase, TrialStatus
from src.models.schemas import ClinicalTrialRecord, TrialAdverseEvent, TrialIntervention

_PHASE_MAP = {
    "1": TrialPhase.PHASE_1,
    "PHASE1": TrialPhase.PHASE_1,
    "EARLYPHASE1": TrialPhase.PHASE_1,
    "2": TrialPhase.PHASE_2,
    "PHASE2": TrialPhase.PHASE_2,
    "3": TrialPhase.PHASE_3,
    "PHASE3": TrialPhase.PHASE_3,
    "4": TrialPhase.PHASE_4,
    "PHASE4": TrialPhase.PHASE_4,
}
_STATUS_MAP = {
    "COMPLETED": TrialStatus.COMPLETED,
    "RECRUITING": TrialStatus.RECRUITING,
    "TERMINATED": TrialStatus.TERMINATED,
    "WITHDRAWN": TrialStatus.WITHDRAWN,
}


def map_phase(raw: Any) -> TrialPhase:
    """Map a phase string/list (e.g. "Phase 1", "PHASE1") to :class:`TrialPhase`."""
    if isinstance(raw, list):
        # Take the highest recognized phase in the list.
        mapped = [map_phase(p) for p in raw]
        recognized = [p for p in mapped if p is not TrialPhase.NA]
        return max(recognized, key=lambda p: p.value) if recognized else TrialPhase.NA
    key = str(raw or "").upper().replace(" ", "").replace("_", "")
    return _PHASE_MAP.get(key, TrialPhase.NA)


def _map_status(raw: Any) -> TrialStatus:
    """Map an API status string to :class:`TrialStatus`."""
    key = str(raw or "").upper().replace(" ", "_")
    return _STATUS_MAP.get(key, TrialStatus.OTHER)


def _from_sample(item: dict) -> ClinicalTrialRecord:
    """Validate an item already shaped like :class:`ClinicalTrialRecord`."""
    return ClinicalTrialRecord.model_validate(item)


def _parse_adverse_events(results_section: dict) -> list[TrialAdverseEvent]:
    """Extract adverse events from an API ``resultsSection.adverseEventsModule``."""
    module = (results_section or {}).get("adverseEventsModule", {}) or {}
    events: list[TrialAdverseEvent] = []
    for group_key in ("seriousEvents", "otherEvents"):
        for event in module.get(group_key, []) or []:
            term = str(event.get("term", "")).strip()
            if not term:
                continue
            affected = at_risk = 0
            for stat in event.get("stats", []) or []:
                affected += int(stat.get("numAffected", 0) or 0)
                at_risk += int(stat.get("numAtRisk", 0) or 0)
            frequency = (affected / at_risk * 100.0) if at_risk else 0.0
            events.append(
                TrialAdverseEvent(
                    term=term,
                    organ_system=str(event.get("organSystem", "")).strip(),
                    affected_count=affected,
                    at_risk_count=at_risk,
                    frequency_percent=round(frequency, 2),
                )
            )
    return events


def _from_api(study: dict) -> ClinicalTrialRecord:
    """Map a raw ClinicalTrials.gov v2 ``study`` into a record."""
    protocol = study.get("protocolSection", {}) or {}
    ident = protocol.get("identificationModule", {}) or {}
    status_mod = protocol.get("statusModule", {}) or {}
    design = protocol.get("designModule", {}) or {}
    conditions_mod = protocol.get("conditionsModule", {}) or {}
    arms_mod = protocol.get("armsInterventionsModule", {}) or {}

    interventions = [
        TrialIntervention(
            name=str(iv.get("name", "")).strip(),
            type=str(iv.get("type", "other")).lower()
            if str(iv.get("type", "")).lower() in {"drug", "biological", "other"}
            else "other",
        )
        for iv in arms_mod.get("interventions", []) or []
        if iv.get("name")
    ]

    return ClinicalTrialRecord(
        nct_id=str(ident.get("nctId", "")).strip(),
        title=str(ident.get("briefTitle") or ident.get("officialTitle") or "").strip(),
        phase=map_phase(design.get("phases")),
        status=_map_status(status_mod.get("overallStatus")),
        conditions=list(conditions_mod.get("conditions", []) or []),
        interventions=interventions,
        adverse_events=_parse_adverse_events(study.get("resultsSection", {})),
        enrollment=(design.get("enrollmentInfo", {}) or {}).get("count"),
        start_date=(status_mod.get("startDateStruct", {}) or {}).get("date"),
        completion_date=(status_mod.get("completionDateStruct", {}) or {}).get("date"),
    )


def load_clinical_trials(data_dir: Path) -> list[ClinicalTrialRecord]:
    """Load and validate all clinical trials from a directory of JSON files.

    Args:
        data_dir: Directory containing trial ``*.json`` files (sample or
            ClinicalTrials.gov v2 format).

    Returns:
        A list of validated :class:`ClinicalTrialRecord` objects (empty if none).
    """
    trials: list[ClinicalTrialRecord] = []
    for filename, document in read_json_documents(Path(data_dir)):
        is_api = isinstance(document, dict) and "studies" in document
        for item in extract_items(document, "studies"):
            if not isinstance(item, dict):
                continue
            try:
                if is_api or "protocolSection" in item:
                    trials.append(_from_api(item))
                else:
                    trials.append(_from_sample(item))
            except (ValidationError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Skipping malformed trial in {}: {}", filename, exc)

    logger.info("Loaded {} clinical trials from {}", len(trials), data_dir)
    return trials

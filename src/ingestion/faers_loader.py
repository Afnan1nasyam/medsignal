"""Loader for FDA FAERS adverse-event reports (local JSON files only).

Parses both our simplified sample format (already shaped like
:class:`FaersReport`) and the openFDA ``/drug/event.json`` API response format
into validated :class:`FaersReport` models. Malformed records are logged and
skipped so a single bad record never aborts a load.
"""

from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from src.ingestion._common import extract_items, first, read_json_documents
from src.models.enums import DrugRole, ReactionOutcome, ReporterType
from src.models.schemas import FaersDrug, FaersPatient, FaersReaction, FaersReport
from src.utils.medical_terms import normalize_drug_name

# openFDA coded-value maps.
_SEX_MAP = {"1": "male", "2": "female"}
_ROLE_MAP = {
    "1": DrugRole.PRIMARY_SUSPECT,
    "2": DrugRole.CONCOMITANT,
    "3": DrugRole.INTERACTING,
}
_REPORTER_MAP = {
    "1": ReporterType.PHYSICIAN,
    "2": ReporterType.PHARMACIST,
    "5": ReporterType.CONSUMER,
}
# report-level seriousness flags → a representative reaction outcome.
_SERIOUS_OUTCOME_FIELDS = [
    ("seriousnessdeath", ReactionOutcome.DEATH),
    ("seriousnesslifethreatening", ReactionOutcome.LIFE_THREATENING),
    ("seriousnesshospitalization", ReactionOutcome.HOSPITALIZATION),
    ("seriousnessdisabling", ReactionOutcome.DISABILITY),
    ("seriousnesscongenitalanomali", ReactionOutcome.CONGENITAL_ANOMALY),
]


def _format_date(raw: str | None) -> str | None:
    """Convert an openFDA ``YYYYMMDD`` date to ``YYYY-MM-DD``."""
    if raw and len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw


def _normalize_drugs(report: FaersReport) -> FaersReport:
    """Populate each drug's generic_name via normalization when missing."""
    for drug in report.drugs:
        if not drug.generic_name:
            drug.generic_name = normalize_drug_name(drug.name)
    return report


def _from_sample(item: dict) -> FaersReport:
    """Validate an item already shaped like :class:`FaersReport`."""
    return _normalize_drugs(FaersReport.model_validate(item))


def _from_openfda(item: dict) -> FaersReport:
    """Map a raw openFDA event ``result`` into a :class:`FaersReport`."""
    patient = item.get("patient", {}) or {}

    faers_patient = FaersPatient(
        age=_to_float(patient.get("patientonsetage")),
        age_unit=patient.get("patientonsetageunit"),
        sex=_SEX_MAP.get(str(patient.get("patientsex", "")), "unknown"),
        weight=_to_float(patient.get("patientweight")),
    )

    drugs: list[FaersDrug] = []
    for raw_drug in patient.get("drug", []) or []:
        name = raw_drug.get("medicinalproduct", "").strip()
        if not name:
            continue
        substance = raw_drug.get("activesubstance", {}) or {}
        drugs.append(
            FaersDrug(
                name=name,
                generic_name=substance.get("activesubstancename")
                or normalize_drug_name(name),
                role=_ROLE_MAP.get(
                    str(raw_drug.get("drugcharacterization", "")), DrugRole.CONCOMITANT
                ),
                indication=raw_drug.get("drugindication"),
                dosage=raw_drug.get("drugdosagetext"),
                route=raw_drug.get("drugadministrationroute"),
            )
        )

    outcome = ReactionOutcome.OTHER
    for field_name, mapped in _SERIOUS_OUTCOME_FIELDS:
        if str(item.get(field_name, "")) == "1":
            outcome = mapped
            break

    reactions = [
        FaersReaction(
            preferred_term=raw_reaction.get("reactionmeddrapt", "").strip(),
            outcome=outcome,
        )
        for raw_reaction in patient.get("reaction", []) or []
        if raw_reaction.get("reactionmeddrapt")
    ]

    qualification = str(
        (item.get("primarysource", {}) or {}).get("qualification", "")
    )

    return FaersReport(
        report_id=str(item.get("safetyreportid", "")).strip(),
        patient=faers_patient,
        drugs=drugs,
        reactions=reactions,
        report_date=_format_date(item.get("receiptdate")),
        reporter_type=_REPORTER_MAP.get(qualification, ReporterType.OTHER),
        serious=str(item.get("serious", "")) == "1",
    )


def _to_float(value: object) -> float | None:
    """Best-effort float coercion, returning None on failure."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def load_faers_reports(data_dir: Path) -> list[FaersReport]:
    """Load and validate all FAERS reports from a directory of JSON files.

    Args:
        data_dir: Directory containing FAERS ``*.json`` files (sample or
            openFDA format).

    Returns:
        A list of validated :class:`FaersReport` objects (empty if none).
    """
    reports: list[FaersReport] = []
    for filename, document in read_json_documents(Path(data_dir)):
        is_openfda = isinstance(document, dict) and "results" in document
        for item in extract_items(document, "results"):
            if not isinstance(item, dict):
                continue
            try:
                if is_openfda or "safetyreportid" in item:
                    reports.append(_from_openfda(item))
                else:
                    reports.append(_from_sample(item))
            except (ValidationError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Skipping malformed FAERS report in {}: {}", filename, exc)

    logger.info("Loaded {} FAERS reports from {}", len(reports), data_dir)
    return reports

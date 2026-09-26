"""Loader for drug labels (local JSON files only).

Parses both our simplified sample format (already shaped like
:class:`DrugLabel`) and the openFDA ``/drug/label.json`` API format
(``results[].openfda`` plus free-text sections) into validated
:class:`DrugLabel` models. Malformed records are logged and skipped.
"""

from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from src.ingestion._common import extract_items, first, read_json_documents
from src.models.enums import AEFrequency, InteractionSeverity
from src.models.schemas import DrugLabel, LabelAdverseReaction, LabelDrugInteraction


def _as_list(value: object) -> list[str]:
    """Coerce a scalar/list/None into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _from_sample(item: dict) -> DrugLabel:
    """Validate an item already shaped like :class:`DrugLabel`."""
    return DrugLabel.model_validate(item)


def _from_openfda(item: dict) -> DrugLabel:
    """Map a raw openFDA label ``result`` into a :class:`DrugLabel`.

    openFDA stores adverse reactions and interactions as free-text blocks, so
    each block is stored as a single entry with UNKNOWN frequency / MODERATE
    severity (structured splitting is only possible for our sample format).
    """
    openfda = item.get("openfda", {}) or {}

    adverse_reactions = [
        LabelAdverseReaction(
            reaction="adverse_reactions",
            frequency=AEFrequency.UNKNOWN,
            description=block,
        )
        for block in _as_list(item.get("adverse_reactions"))
    ]

    drug_interactions = [
        LabelDrugInteraction(
            interacting_drug="see description",
            severity=InteractionSeverity.MODERATE,
            description=block,
        )
        for block in _as_list(item.get("drug_interactions"))
    ]

    return DrugLabel(
        drug_name=str(
            first(openfda.get("brand_name"))
            or first(openfda.get("generic_name"))
            or ""
        ).strip(),
        generic_name=str(first(openfda.get("generic_name"), "")).strip(),
        active_ingredient=str(first(openfda.get("substance_name"), "")).strip(),
        manufacturer=first(openfda.get("manufacturer_name")),
        indications=_as_list(item.get("indications_and_usage")),
        contraindications=_as_list(item.get("contraindications")),
        warnings=_as_list(item.get("warnings") or item.get("warnings_and_cautions")),
        adverse_reactions=adverse_reactions,
        drug_interactions=drug_interactions,
        boxed_warning=first(item.get("boxed_warning")),
    )


def load_drug_labels(data_dir: Path) -> list[DrugLabel]:
    """Load and validate all drug labels from a directory of JSON files.

    Args:
        data_dir: Directory containing label ``*.json`` files (sample or
            openFDA format).

    Returns:
        A list of validated :class:`DrugLabel` objects (empty if none).
    """
    labels: list[DrugLabel] = []
    for filename, document in read_json_documents(Path(data_dir)):
        is_openfda = isinstance(document, dict) and "results" in document
        for item in extract_items(document, "results"):
            if not isinstance(item, dict):
                continue
            try:
                if is_openfda or "openfda" in item:
                    labels.append(_from_openfda(item))
                else:
                    labels.append(_from_sample(item))
            except (ValidationError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Skipping malformed drug label in {}: {}", filename, exc)

    logger.info("Loaded {} drug labels from {}", len(labels), data_dir)
    return labels

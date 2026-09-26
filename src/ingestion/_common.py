"""Shared helpers for the data-source loaders (local file I/O only)."""

import json
from pathlib import Path
from typing import Any

from loguru import logger


def read_json_documents(data_dir: Path) -> list[tuple[str, Any]]:
    """Read every ``*.json`` file in a directory, skipping corrupt files.

    Args:
        data_dir: Directory to scan for JSON files.

    Returns:
        A list of ``(filename, parsed_json)`` tuples. Returns an empty list if
        the directory does not exist. Corrupt/unreadable files are logged and
        skipped rather than raising.
    """
    if not data_dir.exists():
        logger.warning("Data directory does not exist: {}", data_dir)
        return []

    documents: list[tuple[str, Any]] = []
    for path in sorted(data_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                documents.append((path.name, json.load(handle)))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.warning("Skipping corrupt JSON file {}: {}", path.name, exc)
    return documents


def extract_items(document: Any, envelope_key: str) -> list[Any]:
    """Flatten a JSON document into a list of raw record items.

    Handles three shapes: an API envelope ``{envelope_key: [...]}``, a bare
    list of records, or a single record object.

    Args:
        document: The parsed JSON document.
        envelope_key: The key an API envelope stores its records under
            (e.g. ``"results"``, ``"studies"``).

    Returns:
        A list of raw record items (dicts).
    """
    if isinstance(document, dict) and envelope_key in document:
        items = document[envelope_key]
        return list(items) if isinstance(items, list) else [items]
    if isinstance(document, list):
        return document
    if isinstance(document, dict):
        return [document]
    return []


def first(value: Any, default: str | None = None) -> Any:
    """Return the first element of a list, or the value itself, or a default.

    openFDA wraps most scalar fields in single-element lists; this coerces
    those back to scalars.
    """
    if isinstance(value, list):
        return value[0] if value else default
    if value is None:
        return default
    return value

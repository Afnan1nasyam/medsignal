# Requires network access (openFDA). Run on personal laptop — blocked on the dev proxy.
"""Fetch drug labels from the openFDA drug-labeling endpoint.

Saves the raw openFDA response ({"results": [...]}) into ``data/raw/drug_labels/``
so that ``drug_label_loader`` can parse it via its openFDA code path at ingest
time.

Run (personal laptop):
    python scripts/fetch_drug_labels.py --limit 200
    python scripts/fetch_drug_labels.py --search 'openfda.generic_name:metformin'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from loguru import logger

from src.config import settings

_ENDPOINT = "https://api.fda.gov/drug/label.json"
_PAGE = 100


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch drug labels from openFDA.")
    parser.add_argument("--limit", type=int, default=200, help="Total labels to fetch.")
    parser.add_argument("--search", type=str, default=None, help="openFDA search expression.")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default data/raw/drug_labels).")
    parser.add_argument("--api-key", type=str, default=settings.OPENFDA_API_KEY, help="openFDA API key.")
    return parser.parse_args(argv)


def fetch(limit: int, search: str | None, api_key: str) -> list[dict]:
    """Page through openFDA labels and return accumulated results."""
    results: list[dict] = []
    skip = 0
    while len(results) < limit:
        params = {"limit": min(_PAGE, limit - len(results)), "skip": skip}
        if search:
            params["search"] = search
        if api_key:
            params["api_key"] = api_key
        logger.info("openFDA label request: skip={} got={}/{}", skip, len(results), limit)
        response = requests.get(_ENDPOINT, params=params, timeout=60)
        if response.status_code == 404:
            break
        response.raise_for_status()
        batch = response.json().get("results", [])
        if not batch:
            break
        results.extend(batch)
        skip += len(batch)
        time.sleep(0.3)
    return results[:limit]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out) if args.out else settings.data_path / "raw" / "drug_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = fetch(args.limit, args.search, args.api_key)
    out_path = out_dir / "drug_labels_openfda.json"
    out_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    logger.info("Saved {} drug labels to {}", len(results), out_path)
    print(f"Saved {len(results)} drug labels to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

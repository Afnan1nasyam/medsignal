# Requires network access (ClinicalTrials.gov). Run on personal laptop — blocked on the dev proxy.
"""Fetch studies from the ClinicalTrials.gov REST API v2.

Saves the raw API response ({"studies": [...]}) into ``data/raw/clinical_trials/``
so that ``clinical_trials_loader`` can parse it via its v2 code path at ingest
time.

Run (personal laptop):
    python scripts/fetch_clinical_trials.py --limit 300 --query "drug safety"
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

_ENDPOINT = "https://clinicaltrials.gov/api/v2/studies"
_PAGE = 100


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch studies from ClinicalTrials.gov v2.")
    parser.add_argument("--limit", type=int, default=300, help="Total studies to fetch.")
    parser.add_argument("--query", type=str, default="drug safety adverse events", help="Free-text query term.")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default data/raw/clinical_trials).")
    return parser.parse_args(argv)


def fetch(limit: int, query: str) -> list[dict]:
    """Page through the v2 API (pageToken) and return accumulated studies."""
    studies: list[dict] = []
    page_token: str | None = None
    while len(studies) < limit:
        params = {
            "query.term": query,
            "pageSize": min(_PAGE, limit - len(studies)),
            # Include results section so adverse events are available.
            "fields": "protocolSection,resultsSection",
        }
        if page_token:
            params["pageToken"] = page_token
        logger.info("CTgov request: got={}/{}", len(studies), limit)
        response = requests.get(_ENDPOINT, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("studies", [])
        if not batch:
            break
        studies.extend(batch)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.3)
    return studies[:limit]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out) if args.out else settings.data_path / "raw" / "clinical_trials"
    out_dir.mkdir(parents=True, exist_ok=True)

    studies = fetch(args.limit, args.query)
    out_path = out_dir / "clinical_trials_api.json"
    out_path.write_text(json.dumps({"studies": studies}, indent=2), encoding="utf-8")
    logger.info("Saved {} trials to {}", len(studies), out_path)
    print(f"Saved {len(studies)} trials to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

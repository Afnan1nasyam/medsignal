# Requires network access (openFDA). Run on personal laptop — blocked on the dev proxy.
"""Fetch FDA FAERS adverse-event reports from the openFDA API.

Saves the raw openFDA response ({"results": [...]}) into ``data/raw/faers/`` so
that ``faers_loader`` can parse it via its openFDA code path at ingest time.

Run (personal laptop):
    python scripts/fetch_faers.py --limit 2000
    python scripts/fetch_faers.py --search 'patient.drug.openfda.generic_name:metformin'
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

_ENDPOINT = "https://api.fda.gov/drug/event.json"
_PAGE = 100  # openFDA max per request without special access


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch FAERS reports from openFDA.")
    parser.add_argument("--limit", type=int, default=1000, help="Total reports to fetch.")
    parser.add_argument("--search", type=str, default=None, help="openFDA search expression.")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default data/raw/faers).")
    parser.add_argument("--api-key", type=str, default=settings.OPENFDA_API_KEY, help="openFDA API key.")
    return parser.parse_args(argv)


def fetch(limit: int, search: str | None, api_key: str) -> list[dict]:
    """Page through openFDA and return the accumulated ``results`` items."""
    results: list[dict] = []
    skip = 0
    while len(results) < limit:
        params = {"limit": min(_PAGE, limit - len(results)), "skip": skip}
        if search:
            params["search"] = search
        if api_key:
            params["api_key"] = api_key
        logger.info("openFDA request: skip={} got={}/{}", skip, len(results), limit)
        response = requests.get(_ENDPOINT, params=params, timeout=60)
        if response.status_code == 404:
            break  # openFDA returns 404 past the last page
        response.raise_for_status()
        batch = response.json().get("results", [])
        if not batch:
            break
        results.extend(batch)
        skip += len(batch)
        time.sleep(0.3)  # be polite to the API
    return results[:limit]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out) if args.out else settings.data_path / "raw" / "faers"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = fetch(args.limit, args.search, args.api_key)
    out_path = out_dir / "faers_openfda.json"
    out_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    logger.info("Saved {} FAERS reports to {}", len(results), out_path)
    print(f"Saved {len(results)} FAERS reports to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Retrieval / planning evaluation over eval/test_queries.json.

Scores intent classification and drug detection using the offline keyword
planner (always available), and — when the API is reachable — evidence grade
and source coverage from the live agent. Results are printed as a Rich table
and saved to eval/retrieval_results.json.

Run:
    python eval/eval_retrieval.py                 # uses the API if running
    MEDSIGNAL_API=http://localhost:8000 python eval/eval_retrieval.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.retrieval.query_planner import QueryPlanner

EVAL_DIR = Path(__file__).resolve().parent
QUERIES_PATH = EVAL_DIR / "test_queries.json"
RESULTS_PATH = EVAL_DIR / "retrieval_results.json"
API_BASE = os.environ.get("MEDSIGNAL_API", "http://localhost:8000")

# Evidence-grade strength (A strongest). "min" is satisfied when actual >= expected.
_GRADE_STRENGTH = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
_CONSOLE = Console()


def _api_query(query: str) -> dict | None:
    """POST a query to the live API, or return None if unreachable."""
    try:
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            resp = client.post(f"{API_BASE}/api/query", json={"query": query})
        resp.raise_for_status()
        return resp.json().get("result", {})
    except Exception as exc:  # noqa: BLE001
        logger.debug("API query failed ({}); scoring offline", exc)
        return None


def _grade_ok(actual: str | None, expected_min: str) -> bool | None:
    """Whether the actual grade is at least as strong as the expected minimum."""
    if not actual:
        return None
    return _GRADE_STRENGTH.get(actual.upper(), 0) >= _GRADE_STRENGTH.get(expected_min.upper(), 0)


def evaluate() -> dict:
    """Run the retrieval evaluation and return an aggregate summary dict."""
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    planner = QueryPlanner(llm_client=None)
    api_available = _api_query("warmup") is not None

    rows: list[dict] = []
    for case in queries:
        plan = planner.plan_without_llm(case["query"])
        intent_correct = plan.intent.name == case["expected_intent"]

        expected_drugs = {d.lower() for d in case["expected_drugs"]}
        found_drugs = {d.lower() for d in plan.drugs_mentioned}
        drugs_detected = expected_drugs.issubset(found_drugs) if expected_drugs else True

        grade = None
        sources_consulted: list[str] = []
        if api_available:
            result = _api_query(case["query"])
            if result:
                grade = result.get("evidence_grade")
                sources_consulted = [s.upper() for s in result.get("sources_consulted", [])]

        grade_ok = _grade_ok(grade, case["expected_evidence_grade_min"])
        expected_sources = {s.upper() for s in case.get("expected_sources", [])}
        sources_ok = (
            expected_sources.issubset(set(sources_consulted)) if (api_available and expected_sources) else None
        )

        rows.append(
            {
                "query": case["query"],
                "difficulty": case["difficulty"],
                "expected_intent": case["expected_intent"],
                "actual_intent": plan.intent.name,
                "intent_correct": intent_correct,
                "expected_drugs": sorted(expected_drugs),
                "found_drugs": sorted(found_drugs),
                "drugs_detected": drugs_detected,
                "expected_grade_min": case["expected_evidence_grade_min"],
                "actual_grade": grade,
                "grade_ok": grade_ok,
                "expected_sources": sorted(expected_sources),
                "sources_consulted": sorted(sources_consulted),
                "sources_ok": sources_ok,
            }
        )

    summary = _summarize(rows, api_available)
    _render(rows, summary, api_available)
    RESULTS_PATH.write_text(
        json.dumps({"summary": summary, "results": rows}, indent=2), encoding="utf-8"
    )
    _CONSOLE.print(f"[dim]Saved results to {RESULTS_PATH}[/dim]")
    return summary


def _rate(values: list[bool | None]) -> float:
    """Pass rate over non-None boolean values (0.0 if none apply)."""
    applicable = [v for v in values if v is not None]
    return round(100.0 * sum(applicable) / len(applicable), 1) if applicable else 0.0


def _summarize(rows: list[dict], api_available: bool) -> dict:
    """Aggregate pass rates across all scored dimensions."""
    return {
        "total": len(rows),
        "api_available": api_available,
        "intent_accuracy": _rate([r["intent_correct"] for r in rows]),
        "drug_detection": _rate([r["drugs_detected"] for r in rows]),
        "grade_pass_rate": _rate([r["grade_ok"] for r in rows]),
        "source_coverage": _rate([r["sources_ok"] for r in rows]),
    }


def _mark(value: bool | None) -> str:
    """Render a tri-state check for the table."""
    if value is None:
        return "[dim]—[/dim]"
    return "[green]✓[/green]" if value else "[red]✗[/red]"


def _render(rows: list[dict], summary: dict, api_available: bool) -> None:
    """Print the per-query and summary tables."""
    table = Table(title="Retrieval Evaluation", show_lines=False)
    table.add_column("Query", style="cyan", max_width=48, overflow="ellipsis")
    table.add_column("Diff", style="magenta")
    table.add_column("Intent")
    table.add_column("Drugs")
    table.add_column("Grade")
    table.add_column("Sources")
    for r in rows:
        table.add_row(
            r["query"],
            r["difficulty"],
            _mark(r["intent_correct"]),
            _mark(r["drugs_detected"]),
            _mark(r["grade_ok"]),
            _mark(r["sources_ok"]),
        )
    _CONSOLE.print(table)

    if not api_available:
        _CONSOLE.print("[yellow]API not reachable — grade & source columns scored offline as n/a.[/yellow]")

    summary_table = Table(title="Summary", show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Intent accuracy", f"{summary['intent_accuracy']}%")
    summary_table.add_row("Drug detection", f"{summary['drug_detection']}%")
    summary_table.add_row("Grade pass rate", f"{summary['grade_pass_rate']}%")
    summary_table.add_row("Source coverage", f"{summary['source_coverage']}%")
    _CONSOLE.print(summary_table)


if __name__ == "__main__":
    evaluate()

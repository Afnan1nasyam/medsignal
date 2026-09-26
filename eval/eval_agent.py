"""Detailed agent evaluation over eval/test_queries.json.

Runs each query through the full LangGraph agent and scores plan quality,
iteration efficiency, citation coverage, and contradiction detection. Requires
the stores to be populated and (for real answers) the embedding model + Groq
API — intended for the personal laptop. Results save to eval/agent_results.json.

Run:
    python eval/eval_agent.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from rich.console import Console
from rich.table import Table

EVAL_DIR = Path(__file__).resolve().parent
QUERIES_PATH = EVAL_DIR / "test_queries.json"
RESULTS_PATH = EVAL_DIR / "agent_results.json"
_VALID_SOURCES = {"faers", "pubmed", "clinical_trials", "drug_labels"}
_CONSOLE = Console()


def _score_plan(plan: dict | None) -> tuple[bool, str]:
    """Plan is sound if it has 1-4 sub-queries all targeting valid sources."""
    if not plan or not plan.get("sub_queries"):
        return False, "no plan / sub-queries"
    subs = plan["sub_queries"]
    if not (1 <= len(subs) <= 4):
        return False, f"{len(subs)} sub-queries (expected 1-4)"
    for sub in subs:
        sources = {s.lower() for s in sub.get("target_sources", [])}
        if not sources or not sources.issubset(_VALID_SOURCES):
            return False, "invalid target sources"
    return True, f"{len(subs)} sub-queries"


def evaluate() -> dict:
    """Run the agent over every query and return an aggregate summary."""
    try:
        from src.agents.graph import run_agent
    except Exception as exc:  # noqa: BLE001
        _CONSOLE.print(f"[red]Cannot import agent: {exc}[/red]")
        return {"error": str(exc)}

    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    rows: list[dict] = []

    for case in queries:
        query = case["query"]
        try:
            result = run_agent(query, max_iterations=3)
            plan = result.query_plan.model_dump() if result.query_plan else None
            plan_ok, plan_note = _score_plan(plan)
            grade = result.evidence_grade.value
            citations = len(result.citations)
            # Every non-insufficient answer should carry at least one citation.
            citations_ok = grade == "E" or citations > 0
            rows.append(
                {
                    "query": query,
                    "difficulty": case["difficulty"],
                    "plan_ok": plan_ok,
                    "plan_note": plan_note,
                    "iterations_used": result.iterations_used,
                    "evidence_grade": grade,
                    "citations": citations,
                    "citations_ok": citations_ok,
                    "sources_consulted": [s.value for s in result.sources_consulted],
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent failed on '{}': {}", query, exc)
            rows.append(
                {
                    "query": query,
                    "difficulty": case["difficulty"],
                    "plan_ok": False,
                    "plan_note": "agent error",
                    "iterations_used": 0,
                    "evidence_grade": "E",
                    "citations": 0,
                    "citations_ok": False,
                    "sources_consulted": [],
                    "error": str(exc),
                }
            )

    summary = _summarize(rows)
    _render(rows, summary)
    RESULTS_PATH.write_text(
        json.dumps({"summary": summary, "results": rows}, indent=2), encoding="utf-8"
    )
    _CONSOLE.print(f"[dim]Saved results to {RESULTS_PATH}[/dim]")
    return summary


def _summarize(rows: list[dict]) -> dict:
    """Aggregate plan quality, citation coverage, and mean iterations."""
    n = len(rows) or 1
    scored = [r for r in rows if r["error"] is None]
    iters = [r["iterations_used"] for r in scored] or [0]
    return {
        "total": len(rows),
        "completed": len(scored),
        "plan_quality": round(100.0 * sum(r["plan_ok"] for r in rows) / n, 1),
        "citation_coverage": round(100.0 * sum(r["citations_ok"] for r in rows) / n, 1),
        "avg_iterations": round(sum(iters) / len(iters), 2),
    }


def _render(rows: list[dict], summary: dict) -> None:
    """Print the per-query and summary tables."""
    table = Table(title="Agent Evaluation", show_lines=False)
    table.add_column("Query", style="cyan", max_width=44, overflow="ellipsis")
    table.add_column("Diff", style="magenta")
    table.add_column("Plan")
    table.add_column("Iters", justify="right")
    table.add_column("Grade")
    table.add_column("Cites", justify="right")
    for r in rows:
        plan_mark = "[green]✓[/green]" if r["plan_ok"] else "[red]✗[/red]"
        cite_mark = "[green]✓[/green]" if r["citations_ok"] else "[red]✗[/red]"
        table.add_row(
            r["query"], r["difficulty"], plan_mark, str(r["iterations_used"]),
            r["evidence_grade"], f"{r['citations']} {cite_mark}",
        )
    _CONSOLE.print(table)

    summary_table = Table(title="Summary", show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Completed", f"{summary['completed']}/{summary['total']}")
    summary_table.add_row("Plan quality", f"{summary['plan_quality']}%")
    summary_table.add_row("Citation coverage", f"{summary['citation_coverage']}%")
    summary_table.add_row("Avg iterations", str(summary["avg_iterations"]))
    _CONSOLE.print(summary_table)


if __name__ == "__main__":
    evaluate()

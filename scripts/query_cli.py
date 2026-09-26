"""Interactive command-line client for the MedSignal agent.

Type a drug-safety question and receive a graded, cited answer plus the agent
trace (plan, sources consulted, iterations). Type 'quit' or 'exit' to leave.

Requires populated stores and (for full answers) the embedding model + Groq API;
intended for the personal laptop. Run:
    python scripts/query_cli.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.graph import run_agent

_BANNER = "MedSignal — interactive query CLI. Type 'quit' to exit.\n"


def _print_result(result) -> None:
    """Pretty-print an AgentResult to the terminal."""
    grade = result.evidence_grade
    print("\n" + "=" * 70)
    print(f"Evidence grade: {grade.value} ({grade.label})")
    print("-" * 70)
    print(result.answer)
    print("-" * 70)

    if result.citations:
        print(f"Citations ({len(result.citations)}):")
        for i, citation in enumerate(result.citations, start=1):
            print(f"  [{i}] {citation.source.value}:{citation.reference_id} — {citation.title}")

    print("\nAgent trace:")
    if result.query_plan:
        print(f"  intent: {result.query_plan.intent.value}")
        print(f"  drugs:  {', '.join(result.query_plan.drugs_mentioned) or 'none'}")
        for i, sub in enumerate(result.query_plan.sub_queries, start=1):
            print(f"    sub-query {i}: [{', '.join(s.value for s in sub.target_sources)}] {sub.query}")
    print(f"  sources consulted: {', '.join(s.value for s in result.sources_consulted) or 'none'}")
    print(f"  iterations used:   {result.iterations_used}")
    print("=" * 70 + "\n")


def main() -> int:
    """Run the interactive read-eval-print loop."""
    print(_BANNER)
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            return 0
        try:
            result = run_agent(query)
            _print_result(result)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

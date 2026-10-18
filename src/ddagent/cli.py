"""Command line.

ddagent run            # interactive: review each verified finding, then get the memo
ddagent run --auto     # approve all verified findings
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from .graph import DueDiligenceAgent


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ddagent", description="AI Due-Diligence Analyst")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run due diligence on the data room")
    run.add_argument("--auto", action="store_true", help="approve every verified finding")
    run.add_argument("--out", type=Path, help="write the memo to this markdown file")
    args = p.parse_args(argv)

    agent = DueDiligenceAgent()
    tid = str(uuid.uuid4())
    state = agent.start(tid)
    for step in state["trace"]:
        print(f"  ✓ {step['step']:<12} {', '.join(f'{k}={v}' for k, v in step.items() if k != 'step')}")

    verified = [f for f in state["findings"] if f.status == "verified"]
    for f in state["findings"]:
        if f.status == "rejected":
            print(f"  ✗ critic rejected: {f.title} ({f.critic_note})")

    approved = []
    print(f"\n{len(verified)} findings need your review:\n")
    for f in verified:
        print(f"[{f.severity.upper()}] {f.finding_id} {f.title}\n    {f.detail}\n    sources: {', '.join(f.citations)}")
        if args.auto or input("    approve? [Y/n] ").strip().lower() in ("", "y", "yes"):
            approved.append(f.finding_id)
    final = agent.resume(tid, approved, reviewer="cli")
    if args.out:
        args.out.write_text(final["memo"], encoding="utf-8")
        print(f"\nMemo written to {args.out}")
    else:
        print("\n" + final["memo"])


if __name__ == "__main__":
    main()

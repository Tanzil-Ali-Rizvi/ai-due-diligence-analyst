"""Evaluation: red-flag recall/precision, decoy false alarms, metric accuracy, citation validity and the
critic's ability to catch fabricated findings.

    python -m ddagent.evaluation            # write evals/results/
    python -m ddagent.evaluation --gate     # exit 1 if quality gates fail (CI)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import agents
from .graph import DueDiligenceAgent
from .models import Finding

GATES = {
    "red_flag_recall": 0.85,
    "precision": 0.85,
    "decoy_false_alarms": 0,
    "metric_accuracy": 1.0,
    "citation_validity": 1.0,
    "fabrication_catch_rate": 0.85,
}


def matches(f: Finding, rf: dict) -> bool:
    text = f"{f.title} {f.detail}".lower()
    cites_source = any(c.split("::")[0] == rf["source"] for c in f.citations)
    return f.workstream == rf["workstream"] and cites_source and any(k in text for k in rf["keywords"])


def evaluate(agent: DueDiligenceAgent, truth: dict) -> dict:
    t0 = time.perf_counter()
    state = agent.run_auto("eval")
    seconds = round(time.perf_counter() - t0, 2)
    approved = [f for f in state["findings"] if f.status == "approved"]

    hits = {rf["id"]: next((f.finding_id for f in approved if matches(f, rf)), None) for rf in truth["red_flags"]}
    matched_findings = {fid for fid in hits.values() if fid}
    unmatched = [f for f in approved if f.finding_id not in matched_findings]
    severity_ok = sum(
        1
        for rf in truth["red_flags"]
        if hits[rf["id"]] and next(f for f in approved if f.finding_id == hits[rf["id"]]).severity == rf["severity"]
    )
    false_alarms = [
        d["id"]
        for d in truth["decoys"]
        if any(
            d["chunk"] in f.citations or any(k in f"{f.title} {f.detail}".lower() for k in d["keywords"])
            for f in approved
        )
    ]
    computed = {m.name: m.value for m in state["metrics"]}
    metric_ok = {k: abs(computed.get(k, float("nan")) - v) < 0.051 for k, v in truth["metrics"].items()}
    valid_cites = sum(all(agent.dataroom.get(c) for c in f.citations) and bool(f.citations) for f in approved)

    fabricated = [
        Finding(finding_id=f"FAKE-{i}", origin="adversarial", **{k: v for k, v in fx.items() if k != "type"})
        for i, fx in enumerate(truth["fabricated_findings"], start=1)
    ]
    judged = agents.critique(fabricated, agent.dataroom, state["metrics"], agent.llm)
    caught = {fx["type"]: [] for fx in truth["fabricated_findings"]}
    for fx, j in zip(truth["fabricated_findings"], judged, strict=True):
        caught[fx["type"]].append(j.status == "rejected")

    n_rf = len(truth["red_flags"])
    metrics = {
        "red_flag_recall": round(sum(bool(v) for v in hits.values()) / n_rf, 3),
        "precision": round(len(matched_findings) / len(approved), 3) if approved else 0.0,
        "severity_accuracy": round(severity_ok / n_rf, 3),
        "decoy_false_alarms": len(false_alarms),
        "metric_accuracy": round(sum(metric_ok.values()) / len(metric_ok), 3),
        "citation_validity": round(valid_cites / len(approved), 3) if approved else 0.0,
        "fabrication_catch_rate": round(sum(j.status == "rejected" for j in judged) / len(judged), 3),
        "findings_reported": len(approved),
        "runtime_s": seconds,
    }
    return {
        "metrics": metrics,
        "hits": hits,
        "unmatched": [f.title for f in unmatched],
        "false_alarms": false_alarms,
        "metric_check": metric_ok,
        "fabrication": {k: f"{sum(v)}/{len(v)}" for k, v in caught.items()},
        "critic_notes": {j.title: j.critic_note for j in judged},
        "trace": state["trace"],
        "llm": getattr(agent.llm, "name", "offline rules"),
        "usage": getattr(agent.llm, "usage", {}),
        "memo": state["memo"],
    }


def render(result: dict, truth: dict) -> str:
    m = result["metrics"]
    rows = [f"| {k.replace('_', ' ')} | {v} | {GATES.get(k, '')} |" for k, v in m.items()]
    lines = [
        "# Evaluation report",
        "",
        f"Mode: **{result['llm']}**",
        "",
        "| Metric | Value | Gate |",
        "|---|---|---|",
        *rows,
        "",
        "## Planted red flags",
        "",
        "| ID | Expected | Found as |",
        "|---|---|---|",
    ]
    lines += [
        f"| {rf['id']} | {rf['description']} | {result['hits'][rf['id']] or 'MISSED'} |" for rf in truth["red_flags"]
    ]
    lines += ["", "## Fabricated findings caught by the critic", "", "| Fabrication type | Rejected |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in result["fabrication"].items()]
    if result["unmatched"]:
        lines += ["", "Unmatched findings (count against precision): " + "; ".join(result["unmatched"])]
    return "\n".join(lines) + "\n"


def check_gates(metrics: dict) -> list[str]:
    fails = []
    for k, gate in GATES.items():
        ok = metrics[k] <= gate if k == "decoy_false_alarms" else metrics[k] >= gate
        if not ok:
            fails.append(f"{k}={metrics[k]} (gate {'<=' if k == 'decoy_false_alarms' else '>='} {gate})")
    return fails


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--truth", type=Path, default=Path("evals/ground_truth.json"))
    p.add_argument("--out", type=Path, default=Path("evals/results"))
    p.add_argument("--gate", action="store_true")
    args = p.parse_args(argv)

    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    result = evaluate(DueDiligenceAgent(), truth)
    args.out.mkdir(parents=True, exist_ok=True)
    report = render(result, truth)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    (args.out / "results.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "memo"}, indent=2), encoding="utf-8"
    )
    (args.out / "sample_memo.md").write_text(result["memo"], encoding="utf-8")
    print(report)
    if args.gate:
        fails = check_gates(result["metrics"])
        if fails:
            print("QUALITY GATE FAILED:\n  " + "\n  ".join(fails))
            return 1
        print("Quality gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

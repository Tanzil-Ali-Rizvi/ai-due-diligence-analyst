"""Graph behaviour, the critic, human-in-the-loop, and the LLM code paths via a scripted fake LLM."""

import json
import re

from ddagent import agents
from ddagent.evaluation import check_gates, evaluate
from ddagent.graph import DueDiligenceAgent
from ddagent.models import Finding


def test_critic_rejects_fabrications_and_keeps_grounded_findings(dataroom, metrics, truth):
    fakes = [
        Finding(finding_id=f"F{i}", **{k: v for k, v in fx.items() if k != "type"})
        for i, fx in enumerate(truth["fabricated_findings"])
    ]
    judged = agents.critique(fakes, dataroom, metrics)
    by_type = {fx["type"]: j.status for fx, j in zip(truth["fabricated_findings"], judged, strict=True)}
    assert by_type["invented number"] == by_type["fake source"] == by_type["no citation"] == "rejected"
    good = Finding(
        finding_id="G1",
        workstream="commercial",
        severity="high",
        title="Concentration",
        detail="The largest customer is 45.0% of FY2026 revenue.",
        citations=["commercial/customer_revenue_fy2026.csv::0"],
    )
    assert agents.critique([good], dataroom, metrics)[0].status == "verified"


def test_graph_pauses_for_human_review_and_respects_decisions(settings):
    agent = DueDiligenceAgent(settings)
    paused = agent.start("hitl")
    assert "memo" not in paused  # interrupted before the writer
    verified = [f.finding_id for f in paused["findings"] if f.status == "verified"]
    keep = verified[:2]
    final = agent.resume("hitl", keep, reviewer="test")
    statuses = {f.finding_id: f.status for f in final["findings"]}
    assert all(statuses[i] == "approved" for i in keep)
    assert all(statuses[i] == "dismissed" for i in verified[2:])
    dismissed_titles = [f.title for f in final["findings"] if f.status == "dismissed"]
    assert not any(t in final["memo"] for t in dismissed_titles)


def test_recommendation_logic():
    f = lambda s: Finding(finding_id="x", workstream="legal", severity=s, title="t", detail="d")  # noqa: E731
    assert agents.recommendation([f("critical")]).startswith("DO NOT PROCEED")
    assert agents.recommendation([f("high"), f("high")]).startswith("PROCEED ONLY WITH CONDITIONS")
    assert agents.recommendation([f("medium")]).startswith("PROCEED,")


def test_offline_eval_passes_gates(settings, truth):
    result = evaluate(DueDiligenceAgent(settings), truth)
    assert check_gates(result["metrics"]) == []
    assert result["metrics"]["decoy_false_alarms"] == 0


class ScriptedLLM:
    """Fake LLM: answers by prompt type. First analyst pass invents a number to exercise the revision loop."""

    name = "scripted"

    def __init__(self):
        self.calls = []
        self.usage = {}

    def json(self, system: str, user: str) -> dict:
        self.calls.append(system[:40])
        if "due-diligence plan" in system:
            return {
                "plan": [
                    {
                        "workstream": "commercial",
                        "objective": "concentration",
                        "queries": ["revenue by customer largest customer"],
                    }
                ]
            }
        if "sceptical due-diligence analyst" in system:
            if "WORKSTREAM: commercial" not in user:
                return {"findings": []}
            revised = "reviewer rejected" in user
            pct = "45.0" if revised else "70"
            return {
                "findings": [
                    {
                        "title": "Customer concentration",
                        "severity": "high",
                        "detail": f"The largest customer is {pct}% of revenue.",
                        "citations": ["commercial/customer_revenue_fy2026.csv::0"],
                    }
                ]
            }
        if "verify due-diligence findings" in system:
            ids = re.findall(r"FINDING (\S+):", user)
            return {"verdicts": [{"id": i, "supported": True, "reason": "matches the table"} for i in ids]}
        if "executive summary" in system:
            return {"summary": "One high-severity concentration risk."}
        raise AssertionError(system)


def test_llm_mode_plans_revises_after_critic_and_writes_memo(settings):
    llm = ScriptedLLM()
    agent = DueDiligenceAgent(settings, llm=llm)
    state = agent.run_auto("llm")
    steps = [t["step"] for t in state["trace"]]
    assert steps.count("analyst") == 2 and steps.count("critic") == 2  # one revision loop
    approved = [f for f in state["findings"] if f.status == "approved"]
    assert len(approved) == 1 and "45.0%" in approved[0].detail
    assert "One high-severity concentration risk." in state["memo"]
    # the planner's single workstream was topped up with the default plan for the other four
    assert len(state["plan"]) == 5
    json.dumps(state["trace"])  # trace is serialisable for the UI

"""LangGraph workflow.

plan -> research -> analyse -> critique --(rejections, LLM mode, revisions left)--> analyse
                                     \\--> human_review (interrupt) -> write_memo -> END
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import agents
from .config import Settings, get_settings
from .dataroom import DataRoom
from .llm import build_llm
from .models import WORKSTREAMS, Chunk, Finding, Metric, PlanItem
from .tools import all_metrics

COMPANY = "Kirana Ledger Technologies Pvt. Ltd."


class DDState(TypedDict, total=False):
    plan: list[PlanItem]
    evidence: dict[str, list[Chunk]]
    metrics: list[Metric]
    findings: list[Finding]
    revisions: int
    review: dict[str, Any]
    memo: str
    trace: list[dict[str, Any]]


class DueDiligenceAgent:
    def __init__(self, settings: Settings | None = None, llm=None):
        self.settings = s = settings or get_settings()
        self.dataroom = DataRoom(Path(s.dataroom_dir))
        self.llm = llm if llm is not None else build_llm(s)
        self.graph = self._build()

    # ---------------------------------------------------------------- nodes
    def _log(self, state: DDState, step: str, t0: float, **info) -> list[dict]:
        return [*state.get("trace", []), {"step": step, "ms": round((time.perf_counter() - t0) * 1000, 1), **info}]

    def n_plan(self, state: DDState) -> DDState:
        t0 = time.perf_counter()
        items = agents.plan(self.dataroom, self.llm)
        return {
            "plan": items,
            "revisions": 0,
            "trace": self._log(
                state, "planner", t0, workstreams=len(items), queries=sum(len(i.queries) for i in items)
            ),
        }

    def n_research(self, state: DDState) -> DDState:
        t0 = time.perf_counter()
        evidence = agents.research(self.dataroom, state["plan"], self.settings.search_k)
        metrics = all_metrics(self.dataroom)
        return {
            "evidence": evidence,
            "metrics": metrics,
            "trace": self._log(
                state, "researcher", t0, evidence_chunks=sum(map(len, evidence.values())), metrics=len(metrics)
            ),
        }

    def n_analyse(self, state: DDState) -> DDState:
        t0 = time.perf_counter()
        prev = state.get("findings", [])
        rejected = {f.workstream: [] for f in prev}
        for f in prev:
            if f.status == "rejected":
                rejected[f.workstream].append(f)
        findings: list[Finding] = []
        for ws in WORKSTREAMS:
            if prev and not rejected.get(ws):  # on a revision pass, keep workstreams that were fine
                findings += [f for f in prev if f.workstream == ws]
                continue
            findings += agents.analyse(
                ws,
                state["evidence"].get(ws, []),
                self._metrics_for(ws, state["metrics"]),
                self.llm,
                feedback=rejected.get(ws) or None,
                max_evidence=self.settings.max_evidence_per_workstream,
            )
        return {
            "findings": findings,
            "revisions": state.get("revisions", 0) + (1 if prev else 0),
            "trace": self._log(state, "analyst", t0, findings=len(findings)),
        }

    def _metrics_for(self, ws: str, metrics: list[Metric]) -> list[Metric]:
        """Give each analyst only the metrics computed from its own workstream's documents."""
        return [m for m in metrics if any((c := self.dataroom.get(s)) and c.workstream == ws for s in m.sources)]

    def n_critique(self, state: DDState) -> DDState:
        t0 = time.perf_counter()
        checked = agents.critique(state["findings"], self.dataroom, state["metrics"], self.llm)
        return {
            "findings": checked,
            "trace": self._log(
                state,
                "critic",
                t0,
                verified=sum(f.status == "verified" for f in checked),
                rejected=sum(f.status == "rejected" for f in checked),
            ),
        }

    def route_after_critique(self, state: DDState) -> str:
        has_rejections = any(f.status == "rejected" for f in state["findings"])
        if has_rejections and self.llm is not None and state.get("revisions", 0) < self.settings.max_revisions:
            return "analyse"
        return "human_review"

    def n_human_review(self, state: DDState) -> DDState:
        verified = [f for f in state["findings"] if f.status == "verified"]
        decision = interrupt({"findings": [f.model_dump() for f in verified]})
        approved_ids = set(decision.get("approved", [f.finding_id for f in verified]))
        out = []
        for f in state["findings"]:
            if f.status == "verified":
                f = f.model_copy(update={"status": "approved" if f.finding_id in approved_ids else "dismissed"})
            out.append(f)
        return {
            "findings": out,
            "review": decision,
            "trace": [
                *state.get("trace", []),
                {
                    "step": "human_review",
                    "approved": len(approved_ids),
                    "reviewer": decision.get("reviewer", "unknown"),
                },
            ],
        }

    def n_write(self, state: DDState) -> DDState:
        t0 = time.perf_counter()
        approved = [f for f in state["findings"] if f.status == "approved"]
        memo = agents.write_memo(COMPANY, approved, state["metrics"], self.dataroom, self.llm)
        return {"memo": memo, "trace": self._log(state, "memo_writer", t0, approved=len(approved))}

    # ---------------------------------------------------------------- graph
    def _build(self):
        g = StateGraph(DDState)
        g.add_node("plan", self.n_plan)
        g.add_node("research", self.n_research)
        g.add_node("analyse", self.n_analyse)
        g.add_node("critique", self.n_critique)
        g.add_node("human_review", self.n_human_review)
        g.add_node("write_memo", self.n_write)
        g.add_edge(START, "plan")
        g.add_edge("plan", "research")
        g.add_edge("research", "analyse")
        g.add_edge("analyse", "critique")
        g.add_conditional_edges("critique", self.route_after_critique, ["analyse", "human_review"])
        g.add_edge("human_review", "write_memo")
        g.add_edge("write_memo", END)
        serde = JsonPlusSerializer(
            allowed_msgpack_modules=[("ddagent.models", n) for n in ("PlanItem", "Chunk", "Metric", "Finding")]
        )
        return g.compile(checkpointer=InMemorySaver(serde=serde))

    # ---------------------------------------------------------------- API
    def start(self, thread_id: str) -> dict:
        """Run until the human-review pause. Returns the state including findings awaiting approval."""
        cfg = {"configurable": {"thread_id": thread_id}}
        self.graph.invoke({}, cfg)
        return self.graph.get_state(cfg).values

    def stream_start(self, thread_id: str):
        """Like start(), but yields each finished step so a UI can show live progress."""
        cfg = {"configurable": {"thread_id": thread_id}}
        for update in self.graph.stream({}, cfg, stream_mode="updates"):
            for node in update:
                if node != "__interrupt__":
                    yield node
        return self.graph.get_state(cfg).values

    def state(self, thread_id: str) -> dict:
        return self.graph.get_state({"configurable": {"thread_id": thread_id}}).values

    def resume(self, thread_id: str, approved: list[str], reviewer: str = "analyst") -> dict:
        cfg = {"configurable": {"thread_id": thread_id}}
        self.graph.invoke(Command(resume={"approved": approved, "reviewer": reviewer}), cfg)
        return self.graph.get_state(cfg).values

    def run_auto(self, thread_id: str = "auto") -> dict:
        """Non-interactive run that approves every verified finding (used by CLI and evals)."""
        state = self.start(thread_id)
        verified = [f.finding_id for f in state["findings"] if f.status == "verified"]
        return self.resume(thread_id, verified, reviewer="auto")

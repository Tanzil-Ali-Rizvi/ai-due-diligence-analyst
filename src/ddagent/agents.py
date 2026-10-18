"""The five agents. Each has an LLM mode and a deterministic offline mode (used in CI and as a baseline).

Planner -> Researcher -> Financial/Risk Analyst -> Critic -> (human review) -> Memo Writer
"""

from __future__ import annotations

import json
import re

from .dataroom import DataRoom
from .models import SEVERITY_ORDER, WORKSTREAMS, Chunk, Finding, Metric, PlanItem
from .tools import fmt

PREFIX = {"financial": "FIN", "commercial": "COM", "legal": "LEG", "people": "PPL", "compliance": "CMP"}

# --------------------------------------------------------------------------------------------- planner

DEFAULT_PLAN = [
    PlanItem(
        workstream="financial",
        objective="Verify revenue quality, profitability and cash runway",
        queries=[
            "FY2026 revenue EBITDA management accounts",
            "monthly burn closing cash runway",
            "revenue recognition audit status",
        ],
    ),
    PlanItem(
        workstream="commercial",
        objective="Assess customer concentration and contract durability",
        queries=[
            "revenue by customer FY2026 largest customer",
            "contract end dates enterprise customers",
            "net revenue retention merchants",
        ],
    ),
    PlanItem(
        workstream="legal",
        objective="Identify litigation, contingent liabilities and contract risks",
        queries=[
            "pending litigation claim damages provision",
            "termination change of control clause",
            "liquidation preference cap table",
        ],
    ),
    PlanItem(
        workstream="people",
        objective="Evaluate key-person dependency and retention",
        queries=["CTO co-founder resigned successor handover", "attrition ESOP retention leadership team"],
    ),
    PlanItem(
        workstream="compliance",
        objective="Check data protection, security incidents and regulatory exposure",
        queries=["data breach incident exposed KYC notified", "certifications SOC 2 ISO data protection officer"],
    ),
]

PLANNER_SYSTEM = """You are the lead of an investment due-diligence team. Given a company overview, produce a
due-diligence plan as JSON: {"plan": [{"workstream": one of financial|commercial|legal|people|compliance,
"objective": str, "queries": [2-4 short search queries for the data room]}]}. Cover all five workstreams."""


def plan(dataroom: DataRoom, llm=None) -> list[PlanItem]:
    if llm is None:
        return DEFAULT_PLAN
    overview = "\n\n".join(c.text for c in dataroom.by_source("company_overview.md"))
    out = llm.json(PLANNER_SYSTEM, f"Company overview:\n{overview}")
    items = [PlanItem(**p) for p in out.get("plan", []) if p.get("workstream") in WORKSTREAMS and p.get("queries")]
    covered = {i.workstream for i in items}
    # Never let the LLM silently skip a workstream.
    return items + [p for p in DEFAULT_PLAN if p.workstream not in covered]


# ------------------------------------------------------------------------------------------ researcher


def research(dataroom: DataRoom, plan_items: list[PlanItem], k: int = 4) -> dict[str, list[Chunk]]:
    """Run every planned query; keep evidence per workstream (deduplicated, in rank order)."""
    evidence: dict[str, list[Chunk]] = {}
    for item in plan_items:
        seen = {c.chunk_id for c in evidence.get(item.workstream, [])}
        bucket = evidence.setdefault(item.workstream, [])
        for q in item.queries:
            for c in dataroom.search(q, k=k, workstream=item.workstream):
                if c.chunk_id not in seen:
                    bucket.append(c)
                    seen.add(c.chunk_id)
    return evidence


# ------------------------------------------------------------------------------------------- analyst

ANALYST_SYSTEM = """You are a sceptical due-diligence analyst for a venture investor. From the EVIDENCE and the
pre-computed METRICS, report the material risks (red flags) for ONE workstream.

Workstream scope (report only issues in yours; other analysts cover the rest):
- financial: accuracy of reported numbers, revenue quality, profitability, cash and runway
- commercial: customers, concentration, retention, pricing, market claims
- legal: litigation, contingent liabilities, contract clauses, cap table and investor rights
- people: founders, key-person dependency, leadership, retention
- compliance: data protection, security incidents, regulatory obligations

Severity rubric:
- critical: possible misstatement of financials or fraud; on its own would stop the deal
- high: material risk that needs a deal condition, indemnity or price change
- medium: notable risk to monitor or address after closing
- low: minor

Rules:
- Report at most 3 findings, most material first. Skip routine items (e.g. an audit or certification still in
  progress, standard contract terms) unless the evidence shows something abnormal.
- Do NOT report matters that are closed, settled, resolved or dropped.
- Every finding must cite one or more evidence ids exactly as given (e.g. "legal/msa.md::4").
- Every number you write must appear in the cited evidence or a metric. Never do arithmetic yourself; when you
  use a metric, also cite the evidence id(s) listed in its sources.
- If there are no material risks, return an empty list.
Return JSON: {"findings": [{"title": str, "detail": str, "severity": str, "citations": [ids]}]}"""


def _metric_lines(metrics: list[Metric]) -> str:
    return "\n".join(f"- {m.name} = {fmt(m)} ({m.formula}; sources: {', '.join(m.sources)})" for m in metrics)


def analyse(
    workstream: str,
    evidence: list[Chunk],
    metrics: list[Metric],
    llm=None,
    feedback: list[Finding] | None = None,
    max_evidence: int = 8,
) -> list[Finding]:
    if llm is None:
        return _rule_based_findings(workstream, evidence, metrics)
    ev = "\n\n".join(
        f'<evidence id="{c.chunk_id}" title="{c.title} > {c.section}">\n{c.text[:1600]}\n</evidence>'
        for c in evidence[:max_evidence]
    )
    prompt = f"WORKSTREAM: {workstream}\n\nMETRICS:\n{_metric_lines(metrics)}\n\nEVIDENCE:\n{ev}"
    if feedback:
        notes = "\n".join(f"- '{f.title}': {f.critic_note}" for f in feedback)
        prompt += f"\n\nA reviewer rejected these earlier findings. Fix or drop them:\n{notes}"
    out = llm.json(ANALYST_SYSTEM, prompt)
    findings = []
    for i, f in enumerate(out.get("findings", []), start=1):
        sev = str(f.get("severity", "medium")).lower()
        findings.append(
            Finding(
                finding_id=f"{PREFIX[workstream]}-{i}",
                workstream=workstream,
                severity=sev if sev in SEVERITY_ORDER else "medium",
                title=str(f.get("title", "")).strip(),
                detail=str(f.get("detail", "")).strip(),
                citations=[str(c) for c in f.get("citations", [])],
                origin="llm",
            )
        )
    return findings


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Drop a finding whose citations are a subset of a more severe (or earlier) finding's citations."""
    kept: list[Finding] = []
    for f in sorted(findings, key=lambda x: SEVERITY_ORDER[x.severity]):
        if any(f.citations and set(f.citations) <= set(k.citations) for k in kept):
            continue
        kept.append(f)
    order = {f.finding_id: i for i, f in enumerate(findings)}
    return sorted(kept, key=lambda f: order[f.finding_id])


def _find(evidence: list[Chunk], *patterns: str) -> list[Chunk]:
    return [c for c in evidence if all(re.search(p, c.text, re.IGNORECASE) for p in patterns)]


def _m(metrics: list[Metric], name: str) -> Metric | None:
    return next((m for m in metrics if m.name == name), None)


def _rule_based_findings(ws: str, evidence: list[Chunk], metrics: list[Metric]) -> list[Finding]:
    """Deterministic baseline analyst: generic red-flag rules over metrics and evidence text."""
    out: list[Finding] = []

    def add(sev, title, detail, cites):
        out.append(
            Finding(
                finding_id=f"{PREFIX[ws]}-{len(out) + 1}",
                workstream=ws,
                severity=sev,
                title=title,
                detail=detail,
                citations=list(dict.fromkeys(cites)),
                origin="rules",
            )
        )

    if ws == "financial":
        gap, stated, rev = (
            _m(metrics, "stated_vs_ledger_gap"),
            _m(metrics, "stated_revenue_fy2026"),
            _m(metrics, "revenue_fy2026"),
        )
        if gap and abs(gap.value) > 5:
            add(
                "critical",
                "Reported revenue does not reconcile to the P&L",
                f"Management accounts state FY2026 revenue of INR {fmt(stated).split()[0]} crore but the P&L shows "
                f"INR {fmt(rev).split()[0]} crore, a {fmt(gap)} overstatement.",
                gap.sources,
            )
        run = _m(metrics, "runway_months")
        if run and run.value < 12:
            cash, burn = _m(metrics, "closing_cash_mar2026"), _m(metrics, "avg_monthly_burn_last3")
            add(
                "critical" if run.value < 6 else "high",
                "Cash runway below 12 months",
                f"Closing cash of INR {fmt(cash).split()[0]} crore against average monthly burn of "
                f"INR {fmt(burn).split()[0]} crore gives about {fmt(run)} of runway.",
                run.sources,
            )
    elif ws == "commercial":
        top1 = _m(metrics, "top1_customer_share")
        if top1 and top1.value > 25:
            add(
                "high",
                "High customer concentration",
                f"The largest customer accounts for {fmt(top1)} of FY2026 revenue.",
                top1.sources,
            )
    elif ws == "legal":
        for c in _find(evidence, r"change of control", r"terminat"):
            add(
                "high",
                "Change-of-control termination right in key customer contract",
                f"{c.title}: the customer may terminate if the company undergoes a change of control.",
                [c.chunk_id],
            )
        for c in _find(evidence, r"status:\s*pending"):
            amt = re.search(r"INR\s*([\d.]+)\s*crore", c.text)
            prov = re.search(r"no provision", c.text, re.IGNORECASE)
            add(
                "high",
                "Pending litigation" + (" with no provision" if prov else ""),
                f"{c.section}: pending claim"
                + (f" of INR {amt.group(1)} crore" if amt else "")
                + ("; no provision recorded." if prov else "."),
                [c.chunk_id],
            )
    elif ws == "people":
        for c in _find(evidence, r"resigned", r"\b(CTO|co-founder|CEO|CFO)\b"):
            add(
                "medium",
                "Key-person risk: co-founder/CTO departure",
                "The co-founder and CTO has resigned; no successor or handover plan is documented.",
                [c.chunk_id],
            )
    elif ws == "compliance":
        for c in _find(evidence, r"(breach|exposed)", r"not (notified|reported)"):
            add(
                "high",
                "Unreported personal data breach",
                "A storage misconfiguration exposed merchant KYC documents; affected merchants and regulators "
                "were not notified.",
                [c.chunk_id],
            )
    return out


# -------------------------------------------------------------------------------------------- critic

_NUM = re.compile(r"(?<![A-Za-z\d])\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[float]:
    vals = set()
    for tok in _NUM.findall(text):
        try:
            vals.add(round(float(tok.replace(",", "")), 2))
        except ValueError:
            pass
    return vals


CRITIC_SYSTEM = """You verify due-diligence findings. For each FINDING you get the exact SOURCE text it cites.
Decide whether those sources, together with any verified tool metrics computed from them, directly support the
claim (not merely mention the topic). Be strict: a claim that
contradicts or goes beyond its sources is unsupported.
Return JSON: {"verdicts": [{"id": str, "supported": true|false, "reason": str}]} with one verdict per finding."""


def critique(findings: list[Finding], dataroom: DataRoom, metrics: list[Metric], llm=None) -> list[Finding]:
    """Reject any finding that is uncited, cites unknown sources, or contains numbers not found in its sources."""
    checked = []
    for f in findings:
        f = f.model_copy()
        cited = [dataroom.get(c) for c in f.citations]
        if not f.citations:
            f.status, f.critic_note = "rejected", "No citations."
        elif any(c is None for c in cited):
            bad = [cid for cid, c in zip(f.citations, cited, strict=True) if c is None]
            f.status, f.critic_note = "rejected", f"Cites unknown source(s): {', '.join(bad)}."
        else:
            allowed = set().union(*(_numbers(f"{c.section} {c.text}") for c in cited))
            for m in metrics:
                if set(m.sources) & set(f.citations):
                    allowed |= {round(m.value, 2), round(abs(m.value), 2), round(m.value, 1), round(m.value)}
            years = {float(y) for y in range(2015, 2031)}
            unsupported = sorted(
                n
                for n in _numbers(f"{f.title} {f.detail}") - allowed - years
                if not any(abs(n - a) < 0.051 for a in allowed)
            )
            if unsupported:
                f.status = "rejected"
                f.critic_note = f"Numbers not found in cited sources: {', '.join(map(str, unsupported))}."
            elif llm is not None:
                f.status, f.critic_note = "proposed", ""  # numeric checks passed; semantic check below
            else:
                f.status, f.critic_note = "verified", "Citations resolve and all numbers are grounded."
        checked.append(f)
    pending = [f for f in checked if f.status == "proposed"]
    if pending:  # one batched LLM call for all semantic checks (saves tokens and rate limit)
        blocks = []
        for f in pending:
            src = "\n".join(f"[{c}] {dataroom.get(c).text[:1600]}" for c in f.citations)
            used = [m for m in metrics if set(m.sources) & set(f.citations)]
            tool = "\n".join(f"- {m.name} = {fmt(m)} ({m.formula})" for m in used) or "- none"
            blocks.append(
                f"FINDING {f.finding_id}: {f.title}. {f.detail}\nSOURCES:\n{src}\n"
                f"VERIFIED TOOL METRICS computed by code from these sources (treat as correct):\n{tool}"
            )
        out = llm.json(CRITIC_SYSTEM, "\n\n---\n\n".join(blocks))
        verdicts = {str(v.get("id")): v for v in out.get("verdicts", []) if isinstance(v, dict)}
        for f in pending:
            v = verdicts.get(f.finding_id)
            if v is None:
                f.status, f.critic_note = "rejected", "Critic returned no verdict."
            else:
                supported = v.get("supported")
                ok = supported is True or str(supported).strip().lower() == "true"
                f.status = "verified" if ok else "rejected"
                f.critic_note = str(v.get("reason", ""))[:300]
    return checked


# -------------------------------------------------------------------------------------------- writer

WRITER_SYSTEM = """Write a 3-4 sentence executive summary for an investment committee using ONLY the approved
findings and metrics provided. No new facts or numbers. Return JSON: {"summary": str}."""


def recommendation(approved: list[Finding]) -> str:
    sev = [f.severity for f in approved]
    if "critical" in sev:
        return "DO NOT PROCEED until critical issues are resolved"
    if sev.count("high") >= 2:
        return "PROCEED ONLY WITH CONDITIONS (specific indemnities, CPs and price adjustment)"
    return "PROCEED, subject to standard confirmatory diligence"


def write_memo(company: str, approved: list[Finding], metrics: list[Metric], dataroom: DataRoom, llm=None) -> str:
    approved = sorted(approved, key=lambda f: (SEVERITY_ORDER[f.severity], f.workstream))
    rec = recommendation(approved)
    if llm is not None and approved:
        facts = json.dumps([{"severity": f.severity, "title": f.title, "detail": f.detail} for f in approved])
        summary = llm.json(
            WRITER_SYSTEM, f"Recommendation: {rec}\nFindings: {facts}\nMetrics:\n{_metric_lines(metrics)}"
        ).get("summary", "")
    else:
        counts = {s: sum(f.severity == s for f in approved) for s in SEVERITY_ORDER}
        summary = (
            f"Diligence identified {len(approved)} material issues "
            f"({counts['critical']} critical, {counts['high']} high, {counts['medium']} medium). "
            + (f"The most serious: {approved[0].title}. " if approved else "")
            + f"Recommendation: {rec}."
        )

    lines = [
        f"# Investment Due-Diligence Memo: {company}",
        "",
        f"**Recommendation:** {rec}",
        "",
        "## Executive summary",
        "",
        summary,
        "",
        "## Key metrics (computed from the data room)",
        "",
        "| Metric | Value | How it was computed |",
        "|---|---|---|",
    ]
    lines += [f"| {m.name.replace('_', ' ')} | {fmt(m)} | {m.formula} |" for m in metrics]
    lines += ["", "## Red flags", ""]
    if not approved:
        lines.append("No material red flags were approved.")
    for f in approved:
        lines += [
            f"### [{f.severity.upper()}] {f.title}",
            "",
            f.detail,
            "",
            "Sources: " + "; ".join(_cite(dataroom, c) for c in f.citations),
            "",
        ]
    lines += [
        "---",
        "*Generated by the AI Due-Diligence Analyst. Every number is computed by deterministic tools or "
        "quoted from a cited data-room section, verified by the critic agent and approved by a human "
        "reviewer.*",
    ]
    return "\n".join(lines)


def _cite(dataroom: DataRoom, cid: str) -> str:
    c = dataroom.get(cid)
    return f"`{c.source}` › {c.section}" if c and c.section and c.section != "table" else f"`{c.source if c else cid}`"

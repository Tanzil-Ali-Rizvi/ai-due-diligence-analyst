"""Streamlit UI. Run:  streamlit run ui/app.py"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddagent.graph import COMPANY, DueDiligenceAgent  # noqa: E402
from ddagent.llm import LLMError  # noqa: E402
from ddagent.tools import fmt  # noqa: E402

st.set_page_config(page_title="AI Due-Diligence Analyst", page_icon="🔎", layout="wide")

SEV_ICON = {"critical": "🟥", "high": "🟧", "medium": "🟨", "low": "⬜"}
KEY_METRICS = [
    ("revenue_fy2026", "Revenue FY26 (P&L)"),
    ("stated_revenue_fy2026", "Revenue claimed"),
    ("revenue_growth_fy2026", "YoY growth"),
    ("runway_months", "Cash runway"),
    ("top1_customer_share", "Top customer share"),
]


@st.cache_resource(show_spinner=False)
def load_agent() -> DueDiligenceAgent:
    root = Path(__file__).resolve().parents[1]
    agent = DueDiligenceAgent()
    if not agent.dataroom.chunks:  # running from another working directory
        from ddagent.config import Settings

        agent = DueDiligenceAgent(Settings(dataroom_dir=root / "data" / "dataroom"))
    return agent


agent = load_agent()
ss = st.session_state
ss.setdefault("thread", str(uuid.uuid4()))
mode = f"LLM: `{agent.llm.name}`" if agent.llm else "Offline rule-based agents (no LLM key configured)"

with st.sidebar:
    st.header("🗂️ Virtual data room")
    st.caption(COMPANY)
    for src in agent.dataroom.sources():
        with st.expander(src):
            for c in agent.dataroom.by_source(src):
                if c.section and c.section != "table":
                    st.markdown(f"**{c.section}**")
                st.text(c.text)
    st.divider()
    st.caption(f"Mode: {mode}")
    if st.button("↺ Start over", use_container_width=True):
        for k in ("paused", "final"):
            ss.pop(k, None)
        ss.thread = str(uuid.uuid4())
        st.rerun()

st.title("🔎 AI Due-Diligence Analyst")
st.caption(
    "Planner → Researcher → Financial Analyst → Critic → **You** → Memo writer. "
    "Numbers come from deterministic tools; every claim must cite the data room."
)

if "paused" not in ss:
    st.info(
        f"Target: **{COMPANY}**, raising a Series B of INR 120 crore. "
        "Browse the data room on the left, then run the agents."
    )
    if st.button("▶ Run due diligence", type="primary"):
        with st.status("Agents at work ...", expanded=True) as status:
            try:
                ss.paused = agent.start(ss.thread)
            except LLMError as exc:
                status.update(label="LLM unavailable", state="error")
                st.error(f"{exc}. Try again in a minute (free-tier rate limit) or run offline.")
                st.stop()
            for t in ss.paused["trace"]:
                st.write(f"✓ **{t['step']}** · " + ", ".join(f"{k}: {v}" for k, v in t.items() if k != "step"))
            status.update(label="Analysis complete: your review is needed", state="complete", expanded=False)
        st.rerun()
    st.stop()

state = ss.get("final") or ss.paused
metrics = {m.name: m for m in state["metrics"]}
cols = st.columns(len(KEY_METRICS))
for col, (name, label) in zip(cols, KEY_METRICS, strict=True):
    if name in metrics:
        col.metric(label, fmt(metrics[name]))

with st.expander("Agent trace"):
    for t in state["trace"]:
        st.write(f"**{t['step']}** · " + ", ".join(f"{k}: {v}" for k, v in t.items() if k != "step"))

if "final" not in ss:
    st.subheader("👤 Your review")
    st.caption("Untick anything you disagree with. Only approved findings go into the memo.")
    verified = [f for f in ss.paused["findings"] if f.status == "verified"]
    approved = []
    for f in sorted(verified, key=lambda f: ("critical", "high", "medium", "low").index(f.severity)):
        c1, c2 = st.columns([0.06, 0.94])
        if c1.checkbox("approve", value=True, key=f"ok-{f.finding_id}", label_visibility="collapsed"):
            approved.append(f.finding_id)
        with c2:
            st.markdown(f"{SEV_ICON[f.severity]} **{f.title}** · `{f.workstream}` · {f.severity}")
            st.write(f.detail)
            with st.expander("Evidence"):
                for cid in f.citations:
                    ch = agent.dataroom.get(cid)
                    st.caption(f"{ch.source} › {ch.section}")
                    st.text(ch.text)
    rejected = [f for f in ss.paused["findings"] if f.status == "rejected"]
    if rejected:
        with st.expander(f"🛑 {len(rejected)} finding(s) rejected by the critic agent"):
            for f in rejected:
                st.markdown(f"**{f.title}**: {f.detail}  \n_Critic:_ {f.critic_note}")
    if st.button(f"✅ Approve {len(approved)} finding(s) and write memo", type="primary"):
        with st.spinner("Writing memo ..."):
            ss.final = agent.resume(ss.thread, approved, reviewer="streamlit-user")
        st.rerun()
else:
    st.subheader("📄 Investment memo")
    st.download_button("⬇ Download memo (.md)", ss.final["memo"], file_name="due_diligence_memo.md")
    st.markdown(ss.final["memo"])

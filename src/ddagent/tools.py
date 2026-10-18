"""Deterministic financial tools. Agents call these instead of doing arithmetic in the LLM,
so every number in the memo is reproducible and traceable to its source rows."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .models import Metric

PNL = "financials/pnl_fy2024_fy2026.csv"
MONTHLY = "financials/monthly_metrics_fy2026.csv"
CUSTOMERS = "commercial/customer_revenue_fy2026.csv"
MGMT = "financials/management_accounts_summary.md"


def _rows(root: Path, rel: str) -> list[dict[str, str]]:
    with open(root / rel, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cid(rel: str, n: int = 0) -> str:
    return f"{rel}::{n}"


def pnl_metrics(root: Path) -> list[Metric]:
    rows = {r["fiscal_year"]: r for r in _rows(root, PNL)}
    cur, prev = rows["FY2026"], rows["FY2025"]
    rev, rev_prev = float(cur["revenue_inr_cr"]), float(prev["revenue_inr_cr"])
    cogs = float(cur["cogs_inr_cr"])
    src = [_cid(PNL)]
    return [
        Metric(name="revenue_fy2026", value=rev, unit="INR cr", formula="P&L FY2026 revenue", sources=src),
        Metric(
            name="revenue_growth_fy2026",
            value=round((rev / rev_prev - 1) * 100, 1),
            unit="%",
            formula="FY2026 revenue / FY2025 revenue - 1",
            sources=src,
        ),
        Metric(
            name="gross_margin_fy2026",
            value=round((rev - cogs) / rev * 100, 1),
            unit="%",
            formula="(revenue - COGS) / revenue",
            sources=src,
        ),
        Metric(
            name="ebitda_fy2026",
            value=float(cur["ebitda_inr_cr"]),
            unit="INR cr",
            formula="P&L FY2026 EBITDA",
            sources=src,
        ),
    ]


def runway_metrics(root: Path) -> list[Metric]:
    rows = _rows(root, MONTHLY)
    cash = float(rows[-1]["closing_cash_inr_cr"])
    burn = sum(float(r["net_burn_inr_cr"]) for r in rows[-3:]) / 3
    src = [_cid(MONTHLY)]
    return [
        Metric(name="closing_cash_mar2026", value=cash, unit="INR cr", formula="closing cash, March 2026", sources=src),
        Metric(
            name="avg_monthly_burn_last3",
            value=round(burn, 2),
            unit="INR cr",
            formula="mean net burn, Jan-Mar 2026",
            sources=src,
        ),
        Metric(
            name="runway_months",
            value=round(cash / burn, 1),
            unit="months",
            formula="closing cash / average monthly burn (last 3 months)",
            sources=src,
        ),
    ]


def concentration_metrics(root: Path) -> list[Metric]:
    rows = _rows(root, CUSTOMERS)
    revs = [(r["customer"], float(r["revenue_fy2026_inr_cr"])) for r in rows]
    total = sum(v for _, v in revs)
    named = sorted((v for name, v in revs if not name.lower().startswith("long tail")), reverse=True)
    src = [_cid(CUSTOMERS)]
    return [
        Metric(
            name="top1_customer_share",
            value=round(named[0] / total * 100, 1),
            unit="%",
            formula="largest customer revenue / total revenue",
            sources=src,
        ),
        Metric(
            name="top5_customer_share",
            value=round(sum(named[:5]) / total * 100, 1),
            unit="%",
            formula="top 5 customers revenue / total revenue",
            sources=src,
        ),
    ]


def stated_revenue(root: Path) -> float | None:
    text = (root / MGMT).read_text(encoding="utf-8")
    m = re.search(r"Total revenue for FY2026:\s*INR\s*([\d.]+)\s*crore", text)
    return float(m.group(1)) if m else None


def reconciliation_metrics(root: Path, stated_chunk: str = _cid(MGMT, 1)) -> list[Metric]:
    stated = stated_revenue(root)
    ledger = next(m.value for m in pnl_metrics(root) if m.name == "revenue_fy2026")
    if stated is None:
        return []
    return [
        Metric(
            name="stated_revenue_fy2026",
            value=stated,
            unit="INR cr",
            formula="management accounts headline revenue",
            sources=[stated_chunk],
        ),
        Metric(
            name="stated_vs_ledger_gap",
            value=round((stated - ledger) / ledger * 100, 1),
            unit="%",
            formula="(stated revenue - P&L revenue) / P&L revenue",
            sources=[stated_chunk, _cid(PNL)],
        ),
    ]


def all_metrics(dataroom) -> list[Metric]:
    """Compute every metric; `dataroom` is a DataRoom (used to cite the exact source sections)."""
    root = Path(dataroom.root)
    stated_chunk = next(
        (c.chunk_id for c in dataroom.by_source(MGMT) if "Total revenue for FY2026" in c.text), _cid(MGMT)
    )
    return (
        pnl_metrics(root)
        + runway_metrics(root)
        + concentration_metrics(root)
        + reconciliation_metrics(root, stated_chunk)
    )


def fmt(m: Metric) -> str:
    v = f"{m.value:,.1f}" if m.unit != "INR cr" else f"{m.value:,.2f}".rstrip("0").rstrip(".")
    return f"{v} {m.unit}" if m.unit not in {"%"} else f"{v}%"

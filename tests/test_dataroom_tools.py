import pytest


def test_every_document_is_chunked_with_stable_ids(dataroom):
    assert len(dataroom.sources()) == 10
    assert all(cid.startswith(c.source + "::") for cid, c in dataroom.chunks.items())
    assert dataroom.get("legal/msa_metro_mart_retail.md::4").section == "14. Termination"


def test_search_ranks_the_right_clause_first(dataroom):
    top = dataroom.search("change of control termination", k=1)[0]
    assert top.chunk_id == "legal/msa_metro_mart_retail.md::4"


def test_search_can_filter_by_workstream(dataroom):
    hits = dataroom.search("revenue", k=10, workstream="financial")
    assert hits and all(h.workstream == "financial" for h in hits)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("revenue_fy2026", 41.2),
        ("revenue_growth_fy2026", 40.1),
        ("gross_margin_fy2026", 66.0),
        ("runway_months", 8.1),
        ("top1_customer_share", 45.0),
        ("top5_customer_share", 69.0),
        ("stated_vs_ledger_gap", 16.5),
    ],
)
def test_financial_tools_compute_exact_values(metrics, name, expected):
    assert next(m for m in metrics if m.name == name).value == pytest.approx(expected, abs=0.05)


def test_metrics_cite_real_chunks(dataroom, metrics):
    for m in metrics:
        assert m.sources and all(dataroom.get(s) for s in m.sources), m.name
    gap = next(m for m in metrics if m.name == "stated_vs_ledger_gap")
    assert "financials/management_accounts_summary.md::1" in gap.sources

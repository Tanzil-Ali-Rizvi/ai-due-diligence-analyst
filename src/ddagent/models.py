"""Shared data structures."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["critical", "high", "medium", "low"]
Workstream = Literal["financial", "commercial", "legal", "people", "compliance"]
WORKSTREAMS: tuple[str, ...] = ("financial", "commercial", "legal", "people", "compliance")
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


class Chunk(BaseModel):
    """One citable unit of the data room (a markdown section or a whole CSV table)."""

    chunk_id: str  # "<source>::<n>"
    source: str
    title: str
    section: str
    workstream: str
    text: str


class Metric(BaseModel):
    name: str
    value: float
    unit: str
    formula: str
    sources: list[str]  # chunk ids the inputs came from


class Finding(BaseModel):
    finding_id: str
    workstream: str
    severity: Severity
    title: str
    detail: str
    citations: list[str] = Field(default_factory=list)  # chunk ids
    status: Literal["proposed", "verified", "rejected", "approved", "dismissed"] = "proposed"
    critic_note: str = ""
    origin: str = "analyst"


class PlanItem(BaseModel):
    workstream: str
    objective: str
    queries: list[str]

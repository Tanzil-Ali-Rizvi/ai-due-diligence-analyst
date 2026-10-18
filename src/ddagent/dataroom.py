"""Virtual data room: loading, chunking and BM25 search with stable, citable chunk ids."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .models import Chunk

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEAD = re.compile(r"^#{1,3}\s+(.+?)\s*$", re.MULTILINE)
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = frozenset("a an and are as at be by for from has have in is it of on or the to was with this that".split())

# CSVs carry no front matter, so map them to a workstream by folder.
_FOLDER_WORKSTREAM = {
    "financials": "financial",
    "commercial": "commercial",
    "legal": "legal",
    "corporate": "legal",
    "people": "people",
    "compliance": "compliance",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _front_matter(raw: str) -> tuple[dict[str, str], str]:
    m = _FRONT.match(raw)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[m.end() :]


def _csv_to_text(raw: str) -> str:
    rows = list(csv.DictReader(io.StringIO(raw)))
    return "\n".join("; ".join(f"{k}: {v}" for k, v in row.items()) for row in rows)


class DataRoom:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.chunks: dict[str, Chunk] = {}
        self._load()
        self._ids = list(self.chunks)
        self._bm25 = BM25Okapi([tokenize(f"{c.title} {c.section} {c.text}") for c in self.chunks.values()])

    def _load(self) -> None:
        for path in sorted(self.root.rglob("*")):
            if path.suffix not in {".md", ".csv"} or not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            raw = path.read_text(encoding="utf-8")
            folder = rel.split("/")[0] if "/" in rel else ""
            if path.suffix == ".csv":
                title = path.stem.replace("_", " ").title()
                ws = _FOLDER_WORKSTREAM.get(folder, "commercial")
                self._add(rel, 0, title, "table", ws, _csv_to_text(raw))
                continue
            meta, body = _front_matter(raw)
            title = meta.get("title", path.stem)
            ws = meta.get("workstream", _FOLDER_WORKSTREAM.get(folder, "commercial"))
            heads = list(_HEAD.finditer(body))
            n = 0
            for i, h in enumerate(heads):
                end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
                text = body[h.end() : end].strip()
                if text:
                    self._add(rel, n, title, h.group(1).strip(), ws, text)
                    n += 1
            if not heads and body.strip():
                self._add(rel, 0, title, "", ws, body.strip())

    def _add(self, source: str, n: int, title: str, section: str, ws: str, text: str) -> None:
        cid = f"{source}::{n}"
        self.chunks[cid] = Chunk(chunk_id=cid, source=source, title=title, section=section, workstream=ws, text=text)

    # ---------- access ----------
    def get(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def by_source(self, source: str) -> list[Chunk]:
        return [c for c in self.chunks.values() if c.source == source]

    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks.values()})

    def search(self, query: str, k: int = 4, workstream: str | None = None) -> list[Chunk]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(scores, self._ids, strict=True), reverse=True)
        out = []
        for score, cid in ranked:
            if score <= 0:
                break
            c = self.chunks[cid]
            if workstream and c.workstream != workstream:
                continue
            out.append(c)
            if len(out) >= k:
                break
        return out

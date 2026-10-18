import json
from pathlib import Path

import pytest

from ddagent.config import Settings
from ddagent.dataroom import DataRoom
from ddagent.tools import all_metrics

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(dataroom_dir=ROOT / "data" / "dataroom", llm_provider="offline")


@pytest.fixture(scope="session")
def dataroom(settings) -> DataRoom:
    return DataRoom(settings.dataroom_dir)


@pytest.fixture(scope="session")
def metrics(dataroom):
    return all_metrics(dataroom)


@pytest.fixture(scope="session")
def truth() -> dict:
    return json.loads((ROOT / "evals" / "ground_truth.json").read_text(encoding="utf-8"))

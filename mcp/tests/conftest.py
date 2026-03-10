from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

import jax

jax.config.update("jax_platform_name", "cpu")

TESTS_ROOT = Path(__file__).resolve().parents[0]
ROOT = Path(__file__).resolve().parents[2]
MCP_SRC = ROOT / "mcp"
FIXTURES = TESTS_ROOT / "fixtures"
DATA_ROOT = ROOT / "data"
DOE_ROOT = ROOT

if str(MCP_SRC) not in sys.path:
    sys.path.insert(0, str(MCP_SRC))
if str(DOE_ROOT) not in sys.path:
    sys.path.insert(0, str(DOE_ROOT))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_doe_or_fixture_json(
    *,
    fixture_filename: str,
    doe_candidate_paths: tuple[str, ...],
) -> Dict[str, Any]:
    for candidate in doe_candidate_paths:
        data_path = DATA_ROOT / candidate
        if data_path.is_file():
            return _load_json(data_path)
        doe_path = DOE_ROOT / candidate
        if doe_path.is_file():
            return _load_json(doe_path)
    return _load_json(FIXTURES / fixture_filename)


@pytest.fixture
def conditions_fixture() -> Dict[str, Dict[str, float]]:
    return _load_doe_or_fixture_json(
        fixture_filename="conditions_example.json",
        doe_candidate_paths=(
            "data/experiments/example.json",
            "tests/example.json",
        ),
    )


@pytest.fixture
def measurements_fixture() -> Dict[str, Dict[str, Any]]:
    return _load_doe_or_fixture_json(
        fixture_filename="measurements_example.json",
        doe_candidate_paths=(
            "data/experiments/measurements.json",
            "tests/measurements.json",
        ),
    )

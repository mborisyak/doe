from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from mcp_contracts import (
    EstimateDoeParametersRequest,
    ProposeDoeExperimentsRequest,
    REQUIRED_CONDITION_NAMES,
)
from mcp_engine import DoeEngine
from mcp_errors import ToolExecutionError

try:
    from doe.common import CustomODESystem
except ImportError:  # pragma: no cover
    from doe.doe.common import CustomODESystem

TESTS_ROOT = Path(__file__).resolve().parents[0]
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
DOE_ROOT = ROOT
FIXTURES = TESTS_ROOT / "fixtures"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _estimate_payload(
    conditions: Dict[str, Dict[str, float]],
    measurements: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "model_spec": _dynamic_model_spec(),
        "conditions": conditions,
        "measurements": measurements,
        "optimizer": {
            "iterations": 12,
            "rtol": 1e-6,
            "dtype": "float32",
        },
    }


def _propose_payload(conditions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    history_timestamps = {label: [3.0, 6.0, 9.0] for label in conditions}
    return {
        "model_spec": _dynamic_model_spec(),
        "condition_ranges": {
            "A": [0.1, 5.0],
            "B": [0.1, 5.0],
            "E": [0.1, 5.0],
            "temperature": [0.0, 100.0],
        },
        "parameters": {
            "q": 818.4,
            "K_A": 0.42,
            "K_B": 0.67,
        },
        "history": {
            "conditions": conditions,
            "timestamps": history_timestamps,
        },
        "proposal_config": {
            "n_proposals": 2,
            "timestamps": [1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0],
            "iterations": 10,
            "criterion": "D",
            "seed": 12345,
        },
    }


def _dynamic_model_spec() -> Dict[str, Any]:
    for candidate in (
        DATA_ROOT / "secret" / "simple.json",
        DOE_ROOT / "data" / "secret" / "simple.json",
        DOE_ROOT / "tests" / "simple.json",
    ):
        if candidate.is_file():
            return _load_json(candidate)
    return _load_json(FIXTURES / "model_spec_example.json")


def test_engine_estimate_returns_finite_outputs(
    conditions_fixture: Dict[str, Dict[str, float]],
    measurements_fixture: Dict[str, Dict[str, Any]],
) -> None:
    engine = DoeEngine()
    request = EstimateDoeParametersRequest.parse_obj(
        _estimate_payload(conditions_fixture, measurements_fixture)
    )

    response = engine.estimate_parameters(request)

    assert set(response.parameters.keys()) == set(request.model_spec["parameters"].keys())
    assert response.loss_trace
    assert np.all(np.isfinite(np.array(response.loss_trace)))
    assert set(response.predictions.keys()) == set(conditions_fixture.keys())
    for label, series in response.predictions.items():
        assert len(series) == len(measurements_fixture[label]["timestamps"])
        assert np.all(np.isfinite(np.array(series)))


def test_engine_propose_returns_values_in_range(
    conditions_fixture: Dict[str, Dict[str, float]],
) -> None:
    engine = DoeEngine()
    payload = _propose_payload(conditions_fixture)
    request = ProposeDoeExperimentsRequest.parse_obj(payload)

    response = engine.propose_experiments(request)

    assert len(response.proposed_conditions) == request.proposal_config.n_proposals
    assert len(response.encoded_proposals) == request.proposal_config.n_proposals
    assert np.all(np.isfinite(np.array(response.loss_trace)))
    for condition in response.proposed_conditions:
        for field_name in REQUIRED_CONDITION_NAMES:
            low, high = request.condition_ranges[field_name]
            value = float(getattr(condition, field_name))
            assert low <= value <= high


def test_engine_propose_is_deterministic_for_seed(
    conditions_fixture: Dict[str, Dict[str, float]],
) -> None:
    engine = DoeEngine()
    payload = _propose_payload(conditions_fixture)
    request = ProposeDoeExperimentsRequest.parse_obj(payload)

    response_one = engine.propose_experiments(request)
    response_two = engine.propose_experiments(request)

    assert np.allclose(response_one.encoded_proposals, response_two.encoded_proposals)
    assert np.allclose(response_one.loss_trace, response_two.loss_trace)


def test_engine_rejects_empty_model_spec() -> None:
    engine = DoeEngine()

    with pytest.raises(ToolExecutionError) as exc_info:
        engine._create_model({})

    assert exc_info.value.code == "invalid_model_spec"


def test_engine_creates_dynamic_enzyme_model_from_dict() -> None:
    engine = DoeEngine()
    model = engine._create_model(_dynamic_model_spec())

    assert isinstance(model, CustomODESystem)


def test_engine_dynamic_model_creation_is_not_cached() -> None:
    engine = DoeEngine()
    model_spec = _dynamic_model_spec()

    model_one = engine._create_model(model_spec)
    model_two = engine._create_model(_dynamic_model_spec())

    assert model_one is not model_two


def test_engine_rejects_dynamic_model_with_invalid_spec() -> None:
    engine = DoeEngine()
    model_spec = _dynamic_model_spec()
    model_spec["initial_state"].pop("A")

    with pytest.raises(ToolExecutionError) as exc_info:
        engine._create_model(model_spec)

    assert exc_info.value.code == "invalid_model_spec"


def test_engine_dynamic_model_estimate_runs(
    conditions_fixture: Dict[str, Dict[str, float]],
    measurements_fixture: Dict[str, Dict[str, Any]],
) -> None:
    engine = DoeEngine()
    payload = _estimate_payload(conditions_fixture, measurements_fixture)
    request = EstimateDoeParametersRequest.parse_obj(payload)

    response = engine.estimate_parameters(request)

    assert response.parameters
    assert np.all(np.isfinite(np.array(response.loss_trace)))

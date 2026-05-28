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
FIXTURES = TESTS_ROOT / "fixtures"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_condition_ranges() -> Dict[str, Any]:
    import yaml
    with (ROOT / "config" / "config.yaml").open() as f:
        return yaml.safe_load(f)["conditions"]


def _dynamic_model_spec() -> Dict[str, Any]:
    for candidate in (
        DATA_ROOT / "secret" / "simple.json",
        ROOT / "data" / "secret" / "simple.json",
        ROOT / "tests" / "simple.json",
    ):
        if candidate.is_file():
            return _load_json(candidate)
    return _load_json(FIXTURES / "model_spec_example.json")


def _batch(conditions: Dict[str, Any], measurements: Dict[str, Any]) -> Dict[str, Any]:
    return {"experiments": {
        label: {
            "conditions": conditions[label],
            "measurements": {
                "timestamps": measurements[label]["timestamps"],
                "A": measurements[label]["measurements"],
            },
        }
        for label in conditions
    }}


def _estimate_payload(conditions, measurements) -> Dict[str, Any]:
    return {
        "model_spec": _dynamic_model_spec(),
        "batches": {"batch-1": _batch(conditions, measurements)},
        "optimizer": {"iterations": 12, "rtol": 1e-6, "dtype": "float32"},
    }


def _propose_payload(conditions, measurements, history: bool = True) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_spec": _dynamic_model_spec(),
        "parameters": {"q": 818.4, "K_A": 0.42, "K_B": 0.67},
        "proposal_config": {"n_proposals": 2, "iterations": 10, "criterion": "D", "seed": 12345},
    }
    if history:
        payload["history"] = {"batch-1": _batch(conditions, measurements)}
    return payload


# --------------------------------------------------------------------- estimate
def test_engine_estimate_returns_finite_outputs(conditions_fixture, measurements_fixture) -> None:
    engine = DoeEngine()
    # The engine now passes the script's fitted_model body through verbatim:
    # {parameters, fit: {final_loss, iterations}, auxiliary: {loss_trace, predictions, ...}}.
    body = engine.estimate_parameters(
        EstimateDoeParametersRequest.parse_obj(_estimate_payload(conditions_fixture, measurements_fixture))
    )
    assert set(body["parameters"]) == set(_dynamic_model_spec()["parameters"].keys())
    assert np.all(np.isfinite(np.array(body["auxiliary"]["loss_trace"])))
    assert np.isfinite(body["fit"]["final_loss"])
    # nested store shape {batch: {exp: {timestamps, A}}}
    predictions = body["auxiliary"]["predictions"]
    assert set(predictions) == {"batch-1"}
    by_exp = predictions["batch-1"]
    assert set(by_exp) == set(conditions_fixture.keys())
    for label, series in by_exp.items():
        assert len(series["A"]) == len(measurements_fixture[label]["timestamps"])
        assert np.all(np.isfinite(np.array(series["A"])))


def test_engine_estimate_seeds_from_initial(conditions_fixture, measurements_fixture) -> None:
    engine = DoeEngine()
    payload = _estimate_payload(conditions_fixture, measurements_fixture)
    payload["initial_parameters"] = {"q": 818.4, "K_A": 0.42, "K_B": 0.67}
    body = engine.estimate_parameters(EstimateDoeParametersRequest.parse_obj(payload))
    assert body["parameters"]


# --------------------------------------------------------------------- propose
def _proposed_conditions(body):
    """Ordered list of proposed condition dicts from a propose design body."""
    return [exp["conditions"] for exp in body["experiments"].values()]


def test_engine_propose_returns_values_in_range(conditions_fixture, measurements_fixture) -> None:
    engine = DoeEngine()
    # The engine passes the script's design body through: {experiments, auxiliary.expected}.
    body = engine.propose_experiments(
        ProposeDoeExperimentsRequest.parse_obj(_propose_payload(conditions_fixture, measurements_fixture))
    )
    assert len(body["experiments"]) == 2
    assert len(body["auxiliary"]["expected"]) == 2
    ranges = _load_condition_ranges()
    for condition in _proposed_conditions(body):
        for field_name in REQUIRED_CONDITION_NAMES:
            low, high = ranges[field_name]
            assert low <= float(condition[field_name]) <= high


def test_engine_propose_without_history(conditions_fixture, measurements_fixture) -> None:
    engine = DoeEngine()
    body = engine.propose_experiments(
        ProposeDoeExperimentsRequest.parse_obj(_propose_payload(conditions_fixture, measurements_fixture, history=False))
    )
    assert len(body["experiments"]) == 2


def test_engine_propose_is_deterministic_for_seed(conditions_fixture, measurements_fixture) -> None:
    engine = DoeEngine()
    request = ProposeDoeExperimentsRequest.parse_obj(_propose_payload(conditions_fixture, measurements_fixture))
    one = engine.propose_experiments(request)
    two = engine.propose_experiments(request)
    # same seed -> same proposal, up to JAX/XLA float noise across separate processes.
    for c1, c2 in zip(_proposed_conditions(one), _proposed_conditions(two)):
        for field, value in c1.items():
            assert value == pytest.approx(c2[field], rel=1e-3)


# ----------------------------------------------------------- model construction
def test_engine_rejects_empty_model_spec() -> None:
    with pytest.raises(ToolExecutionError) as exc_info:
        DoeEngine()._create_model({})
    assert exc_info.value.code == "invalid_model_spec"


def test_engine_creates_dynamic_enzyme_model_from_dict() -> None:
    assert isinstance(DoeEngine()._create_model(_dynamic_model_spec()), CustomODESystem)


def test_engine_dynamic_model_creation_is_not_cached() -> None:
    engine = DoeEngine()
    assert engine._create_model(_dynamic_model_spec()) is not engine._create_model(_dynamic_model_spec())


def test_engine_rejects_dynamic_model_with_invalid_spec() -> None:
    spec = _dynamic_model_spec()
    spec["initial_state"].pop("A")
    with pytest.raises(ToolExecutionError) as exc_info:
        DoeEngine()._create_model(spec)
    assert exc_info.value.code == "invalid_model_spec"

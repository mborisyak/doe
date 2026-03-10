from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pytest

try:
    from pydantic.v1 import ValidationError
except ImportError:  # pragma: no cover
    from pydantic import ValidationError

from mcp_contracts import (
    EstimateDoeParametersRequest,
    ProposeDoeExperimentsRequest,
)

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
        "conditions": deepcopy(conditions),
        "measurements": deepcopy(measurements),
    }


def _propose_payload(
    conditions: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
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
            "conditions": deepcopy(conditions),
            "timestamps": history_timestamps,
        },
        "proposal_config": {
            "n_proposals": 2,
            "timestamps": [1.0, 5.0, 9.0, 13.0],
            "iterations": 8,
            "criterion": "D",
            "seed": 42,
        },
    }


def _dynamic_model_spec() -> Dict[str, Any]:
    for candidate in (
        DATA_ROOT / "models" / "simple.json",
        DOE_ROOT / "data" / "models" / "simple.json",
        DOE_ROOT / "tests" / "simple.json",
    ):
        if candidate.is_file():
            return _load_json(candidate)
    return _load_json(FIXTURES / "model_spec_example.json")


def test_estimate_request_applies_defaults(
    conditions_fixture: Dict[str, Dict[str, float]],
    measurements_fixture: Dict[str, Dict[str, Any]],
) -> None:
    payload = _estimate_payload(conditions_fixture, measurements_fixture)

    parsed = EstimateDoeParametersRequest.parse_obj(payload)

    assert parsed.optimizer.iterations == 512
    assert parsed.optimizer.dtype == "float32"


def test_estimate_request_rejects_unknown_field(
    conditions_fixture: Dict[str, Dict[str, float]],
    measurements_fixture: Dict[str, Dict[str, Any]],
) -> None:
    payload = _estimate_payload(conditions_fixture, measurements_fixture)
    payload["extra_field"] = "unknown"

    with pytest.raises(ValidationError):
        EstimateDoeParametersRequest.parse_obj(payload)


def test_estimate_request_rejects_legacy_model_identifier(
    conditions_fixture: Dict[str, Dict[str, float]],
    measurements_fixture: Dict[str, Dict[str, Any]],
) -> None:
    payload = _estimate_payload(conditions_fixture, measurements_fixture)
    payload["model_identifier"] = "enzyme_model_from_dict_v1"

    with pytest.raises(ValidationError):
        EstimateDoeParametersRequest.parse_obj(payload)


def test_estimate_request_rejects_legacy_parameter_ranges(
    conditions_fixture: Dict[str, Dict[str, float]],
    measurements_fixture: Dict[str, Dict[str, Any]],
) -> None:
    payload = _estimate_payload(conditions_fixture, measurements_fixture)
    payload["parameter_ranges"] = {
        "q": [100.0, 2000.0],
        "K_A": [0.01, 2.0],
        "K_B": [0.01, 2.0],
    }

    with pytest.raises(ValidationError):
        EstimateDoeParametersRequest.parse_obj(payload)


def test_estimate_request_rejects_missing_model_spec_parameters(
    conditions_fixture: Dict[str, Dict[str, float]],
    measurements_fixture: Dict[str, Dict[str, Any]],
) -> None:
    payload = _estimate_payload(conditions_fixture, measurements_fixture)
    payload["model_spec"].pop("parameters")

    with pytest.raises(ValidationError):
        EstimateDoeParametersRequest.parse_obj(payload)


def test_estimate_request_rejects_non_finite_value(
    conditions_fixture: Dict[str, Dict[str, float]],
    measurements_fixture: Dict[str, Dict[str, Any]],
) -> None:
    payload = _estimate_payload(conditions_fixture, measurements_fixture)
    payload["conditions"]["experiment 1"]["A"] = float("inf")

    with pytest.raises(ValidationError):
        EstimateDoeParametersRequest.parse_obj(payload)


def test_estimate_request_rejects_mismatched_labels(
    conditions_fixture: Dict[str, Dict[str, float]],
    measurements_fixture: Dict[str, Dict[str, Any]],
) -> None:
    payload = _estimate_payload(conditions_fixture, measurements_fixture)
    payload["measurements"]["extra"] = payload["measurements"]["experiment 1"]

    with pytest.raises(ValidationError):
        EstimateDoeParametersRequest.parse_obj(payload)


def test_estimate_request_rejects_unsorted_timestamps(
    conditions_fixture: Dict[str, Dict[str, float]],
    measurements_fixture: Dict[str, Dict[str, Any]],
) -> None:
    payload = _estimate_payload(conditions_fixture, measurements_fixture)
    payload["measurements"]["experiment 1"]["timestamps"] = [3.0, 2.0, 9.0]

    with pytest.raises(ValidationError):
        EstimateDoeParametersRequest.parse_obj(payload)


def test_estimate_request_rejects_mismatched_measurement_lengths(
    conditions_fixture: Dict[str, Dict[str, float]],
    measurements_fixture: Dict[str, Dict[str, Any]],
) -> None:
    payload = _estimate_payload(conditions_fixture, measurements_fixture)
    payload["measurements"]["experiment 1"]["measurements"] = [0.5, 0.4]

    with pytest.raises(ValidationError):
        EstimateDoeParametersRequest.parse_obj(payload)


def test_propose_request_rejects_missing_seed(
    conditions_fixture: Dict[str, Dict[str, float]],
) -> None:
    payload = _propose_payload(conditions_fixture)
    payload["proposal_config"].pop("seed")

    with pytest.raises(ValidationError):
        ProposeDoeExperimentsRequest.parse_obj(payload)


def test_propose_request_rejects_invalid_n_proposals(
    conditions_fixture: Dict[str, Dict[str, float]],
) -> None:
    payload = _propose_payload(conditions_fixture)
    payload["proposal_config"]["n_proposals"] = 0

    with pytest.raises(ValidationError):
        ProposeDoeExperimentsRequest.parse_obj(payload)


def test_propose_request_rejects_invalid_criterion(
    conditions_fixture: Dict[str, Dict[str, float]],
) -> None:
    payload = _propose_payload(conditions_fixture)
    payload["proposal_config"]["criterion"] = "X"

    with pytest.raises(ValidationError):
        ProposeDoeExperimentsRequest.parse_obj(payload)


def test_propose_request_rejects_history_alignment_mismatch(
    conditions_fixture: Dict[str, Dict[str, float]],
) -> None:
    payload = _propose_payload(conditions_fixture)
    payload["history"]["timestamps"].pop(next(iter(payload["history"]["timestamps"])))

    with pytest.raises(ValidationError):
        ProposeDoeExperimentsRequest.parse_obj(payload)


def test_propose_request_rejects_legacy_model_identifier(
    conditions_fixture: Dict[str, Dict[str, float]],
) -> None:
    payload = _propose_payload(conditions_fixture)
    payload["model_identifier"] = "enzyme_model_from_dict_v1"

    with pytest.raises(ValidationError):
        ProposeDoeExperimentsRequest.parse_obj(payload)


def test_propose_request_rejects_legacy_parameter_ranges(
    conditions_fixture: Dict[str, Dict[str, float]],
) -> None:
    payload = _propose_payload(conditions_fixture)
    payload["parameter_ranges"] = {
        "q": [100.0, 2000.0],
        "K_A": [0.01, 2.0],
        "K_B": [0.01, 2.0],
    }

    with pytest.raises(ValidationError):
        ProposeDoeExperimentsRequest.parse_obj(payload)


def test_propose_request_rejects_invalid_model_spec_parameter_bounds(
    conditions_fixture: Dict[str, Dict[str, float]],
) -> None:
    payload = _propose_payload(conditions_fixture)
    payload["model_spec"]["parameters"]["q"] = [2000.0, 100.0]

    with pytest.raises(ValidationError):
        ProposeDoeExperimentsRequest.parse_obj(payload)


def test_estimate_request_requires_model_spec(
    conditions_fixture: Dict[str, Dict[str, float]],
    measurements_fixture: Dict[str, Dict[str, Any]],
) -> None:
    payload = _estimate_payload(conditions_fixture, measurements_fixture)
    payload.pop("model_spec")

    with pytest.raises(ValidationError):
        EstimateDoeParametersRequest.parse_obj(payload)


def test_propose_request_requires_model_spec(
    conditions_fixture: Dict[str, Dict[str, float]],
) -> None:
    payload = _propose_payload(conditions_fixture)
    payload.pop("model_spec")

    with pytest.raises(ValidationError):
        ProposeDoeExperimentsRequest.parse_obj(payload)


def test_estimate_request_is_valid_when_model_spec_is_provided(
    conditions_fixture: Dict[str, Dict[str, float]],
    measurements_fixture: Dict[str, Dict[str, Any]],
) -> None:
    payload = _estimate_payload(conditions_fixture, measurements_fixture)
    payload["model_spec"] = _dynamic_model_spec()

    parsed = EstimateDoeParametersRequest.parse_obj(payload)
    assert parsed.model_spec is not None


def test_propose_request_is_valid_when_model_spec_is_provided(
    conditions_fixture: Dict[str, Dict[str, float]],
) -> None:
    payload = _propose_payload(conditions_fixture)
    payload["model_spec"] = _dynamic_model_spec()

    parsed = ProposeDoeExperimentsRequest.parse_obj(payload)
    assert parsed.model_spec is not None

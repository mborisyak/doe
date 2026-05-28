from __future__ import annotations

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

# Requests carry the MCP-resolved spec + batch records; spec *content* is validated when
# the engine builds the model, not by the contract.


def _spec() -> Dict[str, Any]:
    return {"states": ["A"], "parameters": {"k": [0.0, 10.0]}, "observables": {"A": "A"}}


def _batch() -> Dict[str, Any]:
    return {"experiments": {"exp-1": {
        "conditions": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 10.0},
        "measurements": {"timestamps": [3.0, 6.0, 9.0], "A": [0.5, 0.4, 0.3]},
    }}}


def _estimate_payload() -> Dict[str, Any]:
    return {"model_spec": _spec(), "batches": {"batch-1": _batch()}}


def _propose_payload() -> Dict[str, Any]:
    return {
        "model_spec": _spec(),
        "parameters": {"k": 1.0},
        "history": {"batch-1": _batch()},
        "proposal_config": {"n_proposals": 2, "iterations": 8, "criterion": "D", "seed": 42},
    }


# ----------------------------------------------------------------- estimate
def test_estimate_applies_optimizer_defaults() -> None:
    parsed = EstimateDoeParametersRequest.parse_obj(_estimate_payload())
    assert parsed.optimizer.iterations == 512
    assert parsed.optimizer.dtype == "float32"
    assert parsed.initial_parameters is None


def test_estimate_rejects_unknown_field() -> None:
    payload = _estimate_payload()
    payload["extra"] = 1
    with pytest.raises(ValidationError):
        EstimateDoeParametersRequest.parse_obj(payload)


def test_estimate_rejects_empty_batches() -> None:
    payload = _estimate_payload()
    payload["batches"] = {}
    with pytest.raises(ValidationError):
        EstimateDoeParametersRequest.parse_obj(payload)


@pytest.mark.parametrize("field", ["model_spec", "batches"])
def test_estimate_requires_field(field: str) -> None:
    payload = _estimate_payload()
    payload.pop(field)
    with pytest.raises(ValidationError):
        EstimateDoeParametersRequest.parse_obj(payload)


def test_estimate_accepts_initial_parameters() -> None:
    payload = _estimate_payload()
    payload["initial_parameters"] = {"k": 1.2}
    parsed = EstimateDoeParametersRequest.parse_obj(payload)
    assert parsed.initial_parameters == {"k": 1.2}


# ----------------------------------------------------------------- propose
def test_propose_accepts_no_history() -> None:
    payload = _propose_payload()
    payload.pop("history")
    assert ProposeDoeExperimentsRequest.parse_obj(payload).history is None


def test_propose_accepts_no_parameters() -> None:
    payload = _propose_payload()
    payload.pop("parameters")
    assert ProposeDoeExperimentsRequest.parse_obj(payload).parameters is None


def test_propose_rejects_missing_seed() -> None:
    payload = _propose_payload()
    payload["proposal_config"].pop("seed")
    with pytest.raises(ValidationError):
        ProposeDoeExperimentsRequest.parse_obj(payload)


def test_propose_rejects_invalid_n_proposals() -> None:
    payload = _propose_payload()
    payload["proposal_config"]["n_proposals"] = 0
    with pytest.raises(ValidationError):
        ProposeDoeExperimentsRequest.parse_obj(payload)


def test_propose_rejects_invalid_criterion() -> None:
    payload = _propose_payload()
    payload["proposal_config"]["criterion"] = "X"
    with pytest.raises(ValidationError):
        ProposeDoeExperimentsRequest.parse_obj(payload)


def test_propose_requires_model_spec() -> None:
    payload = _propose_payload()
    payload.pop("model_spec")
    with pytest.raises(ValidationError):
        ProposeDoeExperimentsRequest.parse_obj(payload)

# Response shapes are no longer contract-validated: the scripts own their record body
# (parameters/experiments + fit + auxiliary) and the engine passes it through verbatim.

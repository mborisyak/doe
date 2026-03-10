from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from mcp_contracts import (
    Condition,
    DYNAMIC_MODEL_IDENTIFIER,
    EstimateDoeParametersResponse,
    ProposeDoeExperimentsResponse,
    MetadataofRun,
)
from mcp_errors import ToolExecutionError
from mcp_tools import DoeMcpService

TESTS_ROOT = Path(__file__).resolve().parents[0]
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
DOE_ROOT = ROOT
FIXTURES = TESTS_ROOT / "fixtures"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _dynamic_model_spec() -> Dict[str, Any]:
    for candidate in (
        DATA_ROOT / "models" / "simple.json",
        DOE_ROOT / "data" / "models" / "simple.json",
        DOE_ROOT / "tests" / "simple.json",
    ):
        if candidate.is_file():
            return _load_json(candidate)
    return _load_json(FIXTURES / "model_spec_example.json")


def _estimate_payload() -> Dict[str, Any]:
    return {
        "model_spec": _dynamic_model_spec(),
        "conditions": {
            "experiment 1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 10.0},
            "experiment 2": {"A": 1.0, "B": 1.2, "E": 1.0, "temperature": 50.0},
        },
        "measurements": {
            "experiment 1": {
                "timestamps": [3.0, 6.0, 9.0],
                "measurements": [0.53, 0.41, 0.31],
            },
            "experiment 2": {
                "timestamps": [3.0, 6.0, 9.0],
                "measurements": [0.54, 0.33, 0.23],
            },
        },
    }


def _propose_payload() -> Dict[str, Any]:
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
            "conditions": {
                "experiment 1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 10.0},
                "experiment 2": {"A": 1.0, "B": 1.2, "E": 1.0, "temperature": 50.0},
            },
            "timestamps": {
                "experiment 1": [3.0, 6.0, 9.0],
                "experiment 2": [3.0, 6.0, 9.0],
            },
        },
        "proposal_config": {
            "n_proposals": 2,
            "timestamps": [1.0, 5.0, 9.0, 13.0],
            "iterations": 8,
            "criterion": "D",
            "seed": 42,
        },
    }


class FakeEngine:
    def estimate_parameters(self, request: Any) -> EstimateDoeParametersResponse:
        del request
        return EstimateDoeParametersResponse(
            parameters={"q": 800.0, "K_A": 0.4, "K_B": 0.6},
            loss_trace=[0.2, 0.1, 0.05],
            predictions={
                "experiment 1": [0.54, 0.40, 0.32],
                "experiment 2": [0.55, 0.34, 0.24],
            },
            metadata=MetadataofRun(
                model_identifier=DYNAMIC_MODEL_IDENTIFIER,
                model_version="1.0.0",
                solver={"id": "optax.lbfgs", "configuration": {}},
                units_map={"time": "s", "concentration": "mM"},
                warnings=[],
                diagnostics={},
                deterministic=True,
                seed=None,
            ),
        )

    def propose_experiments(self, request: Any) -> ProposeDoeExperimentsResponse:
        return ProposeDoeExperimentsResponse(
            proposed_conditions=[
                Condition.parse_obj(
                    {"A": 1.2, "B": 2.3, "E": 1.1, "temperature": 35.0}
                ),
                Condition.parse_obj(
                    {"A": 2.2, "B": 1.3, "E": 1.6, "temperature": 55.0}
                ),
            ],
            encoded_proposals=[
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
            ],
            loss_trace=[10.0, 8.0, 7.0],
            metadata=MetadataofRun(
                model_identifier=DYNAMIC_MODEL_IDENTIFIER,
                model_version="1.0.0",
                solver={"id": "fisher.armijo", "configuration": {}},
                units_map={"time": "s", "concentration": "mM"},
                warnings=[],
                diagnostics={},
                deterministic=True,
                seed=request.proposal_config.seed,
            ),
        )


def test_estimate_response_shape_is_stable() -> None:
    service = DoeMcpService(engine=FakeEngine())
    response = service.fit_parameters(_estimate_payload())

    assert response["ok"] is True
    assert set(response.keys()) == {"ok", "data"}
    assert set(response["data"].keys()) == {
        "parameters",
        "loss_trace",
        "predictions",
        "metadata",
    }


def test_propose_response_shape_is_stable() -> None:
    service = DoeMcpService(engine=FakeEngine())
    response = service.propose_doe_experiments(_propose_payload())

    assert response["ok"] is True
    assert set(response.keys()) == {"ok", "data"}
    assert set(response["data"].keys()) == {
        "proposed_conditions",
        "encoded_proposals",
        "loss_trace",
        "metadata",
    }


def test_validation_error_is_structured() -> None:
    service = DoeMcpService(engine=FakeEngine())
    bad_payload = _estimate_payload()
    bad_payload.pop("conditions")

    response = service.fit_parameters(bad_payload)

    assert response["ok"] is False
    assert response["error"]["code"] == "validation_error"
    assert "errors" in response["error"]["details"]


def test_engine_error_is_mapped_to_error_envelope() -> None:
    class FailingEngine(FakeEngine):
        def propose_experiments(self, request: Any) -> ProposeDoeExperimentsResponse:
            del request
            raise ToolExecutionError(
                code="execution_failed",
                message="proposal failed",
                details={"reason": "bad numeric state"},
            )

    service = DoeMcpService(engine=FailingEngine())
    response = service.propose_doe_experiments(_propose_payload())

    assert response["ok"] is False
    assert response["error"] == {
        "code": "execution_failed",
        "message": "proposal failed",
        "details": {"reason": "bad numeric state"},
    }


def test_unexpected_engine_exception_is_sanitized() -> None:
    class FailingEngine(FakeEngine):
        def estimate_parameters(self, request: Any) -> EstimateDoeParametersResponse:
            del request
            raise RuntimeError("boom")

    service = DoeMcpService(engine=FailingEngine())
    response = service.fit_parameters(_estimate_payload())

    assert response["ok"] is False
    assert response["error"]["code"] == "internal_error"
    assert response["error"]["details"]["type"] == "RuntimeError"

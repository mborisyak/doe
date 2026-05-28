from __future__ import annotations

from typing import Any, Dict

from mcp_errors import ToolExecutionError
from mcp_tools import DoeMcpService


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


class FakeEngine:
    # The engine returns the script's record body verbatim (the service just wraps it in an
    # envelope); fit -> fitted_model body, propose -> design body.
    def estimate_parameters(self, request: Any) -> Dict[str, Any]:
        del request
        return {
            "parameters": {"k": 1.5},
            "fit": {"final_loss": 0.05, "iterations": 3},
            "auxiliary": {
                "loss_trace": [0.2, 0.1, 0.05],
                "predictions": {"batch-1": {"exp-1": {"timestamps": [3.0, 6.0, 9.0], "A": [0.5, 0.4, 0.3]}}},
            },
        }

    def propose_experiments(self, request: Any) -> Dict[str, Any]:
        del request
        return {
            "experiments": {"exp-1": {"conditions": {"A": 1.2, "B": 2.3, "E": 1.1, "temperature": 35.0}}},
            "auxiliary": {"expected": {"exp-1": {"timestamps": [3.0, 6.0, 9.0], "A": [0.5, 0.4, 0.3]}}},
        }


def test_estimate_response_shape_is_stable() -> None:
    service = DoeMcpService(engine=FakeEngine())
    response = service.fit_parameters(_estimate_payload())
    assert response["ok"] is True
    assert set(response.keys()) == {"ok", "data"}
    assert set(response["data"].keys()) == {"parameters", "fit", "auxiliary"}


def test_propose_response_shape_is_stable() -> None:
    service = DoeMcpService(engine=FakeEngine())
    response = service.propose_doe_experiments(_propose_payload())
    assert response["ok"] is True
    assert set(response["data"].keys()) == {"experiments", "auxiliary"}


def test_validation_error_is_structured() -> None:
    service = DoeMcpService(engine=FakeEngine())
    bad_payload = _estimate_payload()
    bad_payload.pop("batches")
    response = service.fit_parameters(bad_payload)
    assert response["ok"] is False
    assert response["error"]["code"] == "validation_error"
    assert "errors" in response["error"]["details"]


def test_engine_error_is_mapped_to_error_envelope() -> None:
    class FailingEngine(FakeEngine):
        def propose_experiments(self, request: Any) -> Dict[str, Any]:
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
        def estimate_parameters(self, request: Any) -> Dict[str, Any]:
            del request
            raise RuntimeError("boom")

    service = DoeMcpService(engine=FailingEngine())
    response = service.fit_parameters(_estimate_payload())
    assert response["ok"] is False
    assert response["error"]["code"] == "internal_error"
    assert response["error"]["details"]["type"] == "RuntimeError"

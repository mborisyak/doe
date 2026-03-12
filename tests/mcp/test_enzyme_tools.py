from __future__ import annotations

from typing import Any

from mcp_contracts import (
    ExperimentTrajectory,
    MetadataofRun,
    SimulateEnzymeDynamicsResponse,
)
from mcp_errors import ToolExecutionError
from mcp_tools import EnzymeMcpService


def _simulate_payload() -> dict[str, Any]:
    return {
        "contract_version": "1.0",
        "conditions": {"exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}},
    }


class FakeRunner:
    def __init__(self) -> None:
        self.sim_requests: list[Any] = []

    def simulate(self, request: Any) -> SimulateEnzymeDynamicsResponse:
        self.sim_requests.append(request)
        return SimulateEnzymeDynamicsResponse(
            experiments={
                "exp-1": ExperimentTrajectory(
                    time_points=[1.0, 2.0],
                    state_trajectories={"A_measured": [0.9, 0.8]},
                )
            },
            metadata=MetadataofRun(
                model_identifier="enzyme",
                model_version="1.0.0",
                solver={"id": "lsoda", "configuration": {}},
                units_map={"time": "s", "concentration": "mM"},
                warnings=[],
                diagnostics={},
                deterministic=True,
                seed=None,
            ),
        )


def test_simulate_response_shape_is_stable() -> None:
    service = EnzymeMcpService(runner=FakeRunner())
    response = service.simulate_enzyme_dynamics(_simulate_payload())

    assert set(response.keys()) == {"ok", "data"}
    assert response["ok"] is True
    assert set(response["data"].keys()) == {
        "contract_version",
        "experiments",
        "metadata",
    }
    assert set(response["data"]["experiments"]["exp-1"].keys()) == {
        "time_points",
        "state_trajectories",
    }


def test_deterministic_behavior_for_fixed_input() -> None:
    runner = FakeRunner()
    service = EnzymeMcpService(runner=runner)
    payload = _simulate_payload()

    response_one = service.simulate_enzyme_dynamics(payload)
    response_two = service.simulate_enzyme_dynamics(payload)

    assert response_one == response_two


def test_validation_error_is_returned_as_structured_error() -> None:
    service = EnzymeMcpService(runner=FakeRunner())
    bad_payload = _simulate_payload()
    bad_payload["conditions"]["exp-1"]["A"] = -1.0

    response = service.simulate_enzyme_dynamics(bad_payload)

    assert response["ok"] is False
    assert response["error"]["code"] == "validation_error"
    assert "errors" in response["error"]["details"]


def test_extra_fields_return_validation_error() -> None:
    service = EnzymeMcpService(runner=FakeRunner())
    payload = _simulate_payload()
    payload["extra"] = {"not": "allowed"}

    response = service.simulate_enzyme_dynamics(payload)

    assert response["ok"] is False
    assert response["error"]["code"] == "validation_error"


def test_internal_execution_error_is_mapped_to_structured_error() -> None:
    class FailingRunner(FakeRunner):
        def simulate(self, request: Any) -> SimulateEnzymeDynamicsResponse:
            raise ToolExecutionError(
                code="cli_execution_failed",
                message="sim failed",
                details={"exit_code": 1},
            )

    service = EnzymeMcpService(runner=FailingRunner())
    response = service.simulate_enzyme_dynamics(_simulate_payload())

    assert response["ok"] is False
    assert response["error"] == {
        "code": "cli_execution_failed",
        "message": "sim failed",
        "details": {"exit_code": 1},
    }

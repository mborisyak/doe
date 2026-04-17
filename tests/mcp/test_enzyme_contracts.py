from __future__ import annotations

import pytest

try:
    from pydantic.v1 import ValidationError
except ImportError:  # pragma: no cover
    from pydantic import ValidationError

from mcp_contracts import SimulateEnzymeDynamicsRequest


def test_simulate_request_rejects_negative_concentration_inputs() -> None:
    with pytest.raises(ValidationError):
        SimulateEnzymeDynamicsRequest.parse_obj(
            {
                "conditions": {
                    "exp-1": {"A": -1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}
                }
            }
        )


def test_simulate_request_rejects_time_field() -> None:
    with pytest.raises(ValidationError):
        SimulateEnzymeDynamicsRequest.parse_obj(
            {
                "conditions": {
                    "exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}
                },
                "time": {"t_start": 10.0, "t_end": 10.0, "measurements": 10},
            }
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("solutions", {"A": 3.0, "B": 3.0, "E": 0.003}),
        (
            "units",
            {
                "concentration": "mM",
                "temperature": "Celsius",
                "time": "s",
                "solution_volume": "mL",
            },
        ),
        ("device", "cpu"),
        ("contract_version", "1.0"),
    ],
)
def test_simulate_request_rejects_removed_fields(field: str, value: object) -> None:
    payload = {
        "conditions": {
            "exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}
        },
        field: value,
    }
    with pytest.raises(ValidationError):
        SimulateEnzymeDynamicsRequest.parse_obj(payload)


def test_simulate_request_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SimulateEnzymeDynamicsRequest.parse_obj(
            {
                "conditions": {
                    "exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}
                },
                "unexpected": "field",
            }
        )


def test_simulate_request_rejects_seed_and_noise_fields() -> None:
    with pytest.raises(ValidationError):
        SimulateEnzymeDynamicsRequest.parse_obj(
            {
                "conditions": {
                    "exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}
                },
                "seed": 7,
            }
        )

    with pytest.raises(ValidationError):
        SimulateEnzymeDynamicsRequest.parse_obj(
            {
                "conditions": {
                    "exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}
                },
                "noise_std": 0.1,
            }
        )


def test_simulate_request_rejects_stringified_numbers() -> None:
    with pytest.raises(ValidationError):
        SimulateEnzymeDynamicsRequest.parse_obj(
            {
                "conditions": {
                    "exp-1": {"A": "1.0", "B": 2.0, "E": 1.0, "temperature": 37.0}
                }
            }
        )

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


def test_simulate_request_rejects_invalid_time_window() -> None:
    with pytest.raises(ValidationError):
        SimulateEnzymeDynamicsRequest.parse_obj(
            {
                "conditions": {
                    "exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}
                },
                "time": {"t_start": 10.0, "t_end": 10.0, "measurements": 10},
            }
        )


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

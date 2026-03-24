from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mcp_contracts import SimulateEnzymeDynamicsRequest
from mcp_errors import ToolExecutionError
from mcp_simulation import EnzymeCliRunner


def _build_runner(root: Path) -> EnzymeCliRunner:
    return EnzymeCliRunner(enzyme_root=root)


def _request() -> SimulateEnzymeDynamicsRequest:
    return SimulateEnzymeDynamicsRequest.parse_obj(
        {
            "conditions": {
                "exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}
            },
        }
    )


def test_runner_simulate_forwards_conditions_to_cli(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    runner = _build_runner(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run(command: list[str]) -> None:
        captured["command"] = command
        conditions_path = Path(command[command.index("--conditions") + 1])
        output_path = Path(command[command.index("--output") + 1])
        captured["conditions"] = json.loads(conditions_path.read_text(encoding="utf-8"))
        output = {
            "exp-1": {
                "timestamps": [1.0, 2.0, 3.0],
                "measurements": [0.5, 0.25, 0.1],
            }
        }
        output_path.write_text(json.dumps(output), encoding="utf-8")

    monkeypatch.setattr(runner, "_run_command", fake_run)

    response = runner.simulate(_request())

    assert captured["conditions"] == {
        "exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}
    }
    assert "--config" not in captured["command"]
    assert "--parameters" not in captured["command"]
    assert response.experiments["exp-1"].timestamps == [1.0, 2.0, 3.0]
    assert response.experiments["exp-1"].A == [
        0.5,
        0.25,
        0.1,
    ]


def test_runner_simulate_rejects_invalid_cli_payload(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    runner = _build_runner(tmp_path)

    def fake_run(command: list[str]) -> None:
        output_path = Path(command[command.index("--output") + 1])
        output = {"exp-1": {"measurements": [0.5, 0.25]}}
        output_path.write_text(json.dumps(output), encoding="utf-8")

    monkeypatch.setattr(runner, "_run_command", fake_run)

    with pytest.raises(ToolExecutionError) as exc:
        runner.simulate(_request())

    assert exc.value.code == "invalid_cli_output"


def test_runner_simulate_does_not_load_config_files(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    runner = _build_runner(tmp_path)

    def fake_run(command: list[str]) -> None:
        output_path = Path(command[command.index("--output") + 1])
        output = {
            "exp-1": {
                "timestamps": [1.0, 2.0],
                "measurements": [0.5, 0.25],
            }
        }
        output_path.write_text(json.dumps(output), encoding="utf-8")

    monkeypatch.setattr(runner, "_run_command", fake_run)

    response = runner.simulate(_request())

    assert response.experiments["exp-1"].timestamps == [1.0, 2.0]


def test_runner_simulate_omits_seed_flag_when_seed_is_none(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    runner = _build_runner(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run(command: list[str]) -> None:
        captured["command"] = command
        output_path = Path(command[command.index("--output") + 1])
        output = {
            "exp-1": {
                "timestamps": [1.0, 2.0],
                "measurements": [0.5, 0.25],
            }
        }
        output_path.write_text(json.dumps(output), encoding="utf-8")

    monkeypatch.setattr(runner, "_run_command", fake_run)

    runner.simulate(_request())

    assert "--seed" not in captured["command"]

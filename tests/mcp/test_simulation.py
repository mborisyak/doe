from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from mcp_contracts import SimulateEnzymeDynamicsRequest
from mcp_errors import ToolExecutionError
from mcp_simulation import EnzymeCliRunner, REQUIRED_PARAMETER_NAMES


def _write_static_parameters_file(root: Path) -> Path:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    parameters_path = config_dir / "parameters-123456789.yaml"
    parameters = {name: 1.0 for name in REQUIRED_PARAMETER_NAMES}
    with parameters_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(parameters, stream, sort_keys=True)
    model_config_path = config_dir / "config.yaml"
    with model_config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump({"noise": 0.123}, stream, sort_keys=True)
    return parameters_path


def test_runner_simulate_transforms_cli_output(
    monkeypatch: Any, tmp_path: Path
) -> None:
    static_parameters_path = _write_static_parameters_file(tmp_path)
    runner = EnzymeCliRunner(enzyme_root=tmp_path)
    captured: dict[str, Any] = {}

    def fake_run(command: list[str]) -> None:
        output_path = Path(command[command.index("--output") + 1])
        config_path = Path(command[command.index("--config") + 1])
        parameters_path = Path(command[command.index("--parameters") + 1])
        captured["config"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        captured["parameters_path"] = parameters_path
        output = {
            "exp-1": {
                "timestamps": [1.0, 2.0, 3.0, 4.0],
                "measurements": [0.5, 0.25, 0.2, 0.1],
            }
        }
        output_path.write_text(json.dumps(output), encoding="utf-8")

    monkeypatch.setattr(runner, "_run_command", fake_run)

    request = SimulateEnzymeDynamicsRequest.parse_obj(
        {
            "conditions": {
                "exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}
            },
            "time": {"t_start": 5.0, "t_end": 15.0, "measurements": 4},
        }
    )

    response = runner.simulate(request)

    assert captured["config"]["experiment"]["duration"] == 10.0
    assert captured["config"]["experiment"]["measurements"] == 5
    assert captured["config"]["noise"] == 0.123
    assert captured["parameters_path"] == static_parameters_path
    assert response.experiments["exp-1"].time_points == [6.0, 7.0, 8.0, 9.0]
    assert response.experiments["exp-1"].state_trajectories["A_measured"] == [
        0.5,
        0.25,
        0.2,
        0.1,
    ]
    assert response.metadata.warnings

def test_runner_simulate_rejects_invalid_cli_payload(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    _write_static_parameters_file(tmp_path)
    runner = EnzymeCliRunner(enzyme_root=tmp_path)

    def fake_run(command: list[str]) -> None:
        output_path = Path(command[command.index("--output") + 1])
        output = {"exp-1": {"measurements": [0.5, 0.25]}}
        output_path.write_text(json.dumps(output), encoding="utf-8")

    monkeypatch.setattr(runner, "_run_command", fake_run)

    request = SimulateEnzymeDynamicsRequest.parse_obj(
        {
            "conditions": {
                "exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}
            },
            "time": {"t_start": 0.0, "t_end": 15.0, "measurements": 2},
        }
    )

    with pytest.raises(ToolExecutionError) as exc:
        runner.simulate(request)

    assert exc.value.code == "invalid_cli_output"


def test_runner_simulate_rejects_missing_static_parameter_file(tmp_path: Path) -> None:
    runner = EnzymeCliRunner(enzyme_root=tmp_path)

    request = SimulateEnzymeDynamicsRequest.parse_obj(
        {
            "conditions": {
                "exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}
            },
            "time": {"t_start": 0.0, "t_end": 15.0, "measurements": 2},
        }
    )

    with pytest.raises(ToolExecutionError) as exc:
        runner.simulate(request)

    assert exc.value.code == "parameter_file_missing"


def test_runner_simulate_omits_seed_flag_when_seed_is_none(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    _write_static_parameters_file(tmp_path)
    runner = EnzymeCliRunner(enzyme_root=tmp_path)
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

    request = SimulateEnzymeDynamicsRequest.parse_obj(
        {
            "conditions": {
                "exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}
            },
            "time": {"t_start": 0.0, "t_end": 15.0, "measurements": 2},
        }
    )

    runner.simulate(request)

    assert "--seed" not in captured["command"]


def test_runner_simulate_rejects_missing_model_config(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    with (config_dir / "parameters-123456789.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump({name: 1.0 for name in REQUIRED_PARAMETER_NAMES}, stream)

    runner = EnzymeCliRunner(enzyme_root=tmp_path)
    request = SimulateEnzymeDynamicsRequest.parse_obj(
        {
            "conditions": {
                "exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}
            },
            "time": {"t_start": 0.0, "t_end": 15.0, "measurements": 2},
        }
    )

    with pytest.raises(ToolExecutionError) as exc:
        runner.simulate(request)

    assert exc.value.code == "model_config_missing"

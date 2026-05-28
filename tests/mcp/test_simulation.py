from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from mcp_contracts import SimulateEnzymeDynamicsRequest
from mcp_errors import ToolExecutionError
from mcp_simulation import EnzymeCliRunner, REQUIRED_PARAMETER_NAMES


def _write_parameters(root: Path) -> Path:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    parameters_path = config_dir / "parameters-123456789.yaml"
    with parameters_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump({name: 1.0 for name in REQUIRED_PARAMETER_NAMES}, stream, sort_keys=True)
    return parameters_path


def _write_enzyme_config(
    root: Path, *, duration: float = 10.0, measurements: int = 4, noise: float = 0.123
) -> None:
    # The sampling window, concentrations, and noise are config-owned now -- the runner reads
    # them from config/enzyme.yaml, not from the request.
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    with (config_dir / "enzyme.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {
                "experiment": {"duration": duration, "measurements": measurements},
                "solutions": {"A": 3.0, "B": 3.0, "E": 3.0e-3},
                "noise": noise,
            },
            stream,
            sort_keys=True,
        )


def _request() -> SimulateEnzymeDynamicsRequest:
    return SimulateEnzymeDynamicsRequest.parse_obj(
        {"conditions": {"exp-1": {"A": 1.0, "B": 2.0, "E": 1.0, "temperature": 37.0}}}
    )


def test_runner_simulate_transforms_cli_output(
    monkeypatch: Any, tmp_path: Path
) -> None:
    static_parameters_path = _write_parameters(tmp_path)
    _write_enzyme_config(tmp_path, duration=10.0, measurements=4, noise=0.123)
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

    response = runner.simulate(_request())

    # the runner sourced the experiment window + concentrations + noise from the config file
    assert captured["config"]["experiment"]["duration"] == 10.0
    assert captured["config"]["experiment"]["measurements"] == 4
    assert captured["config"]["solutions"] == {"A": 3.0, "B": 3.0, "E": 3.0e-3}
    assert captured["config"]["noise"] == 0.123
    assert captured["parameters_path"] == static_parameters_path
    # CLI integrates from t=0 and the window starts at 0 -> timestamps used as-is, no shift
    assert response.experiments["exp-1"].time_points == [1.0, 2.0, 3.0, 4.0]
    assert response.experiments["exp-1"].state_trajectories["A_measured"] == [
        0.5,
        0.25,
        0.2,
        0.1,
    ]
    assert response.metadata.warnings  # noise > 0 -> the noise warning


def test_runner_simulate_rejects_invalid_cli_payload(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    _write_parameters(tmp_path)
    _write_enzyme_config(tmp_path, measurements=2)
    runner = EnzymeCliRunner(enzyme_root=tmp_path)

    def fake_run(command: list[str]) -> None:
        output_path = Path(command[command.index("--output") + 1])
        output = {"exp-1": {"measurements": [0.5, 0.25]}}
        output_path.write_text(json.dumps(output), encoding="utf-8")

    monkeypatch.setattr(runner, "_run_command", fake_run)

    with pytest.raises(ToolExecutionError) as exc:
        runner.simulate(_request())

    assert exc.value.code == "invalid_cli_output"


def test_runner_simulate_rejects_missing_static_parameter_file(tmp_path: Path) -> None:
    _write_enzyme_config(tmp_path)
    runner = EnzymeCliRunner(enzyme_root=tmp_path)

    with pytest.raises(ToolExecutionError) as exc:
        runner.simulate(_request())

    assert exc.value.code == "parameter_file_missing"


def test_runner_simulate_omits_seed_flag(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    _write_parameters(tmp_path)
    _write_enzyme_config(tmp_path, measurements=2)
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

    runner.simulate(_request())

    assert "--seed" not in captured["command"]


def test_runner_simulate_rejects_missing_model_config(
    tmp_path: Path,
) -> None:
    # parameters present but no config/enzyme.yaml -> the config read fails.
    _write_parameters(tmp_path)
    runner = EnzymeCliRunner(enzyme_root=tmp_path)

    with pytest.raises(ToolExecutionError) as exc:
        runner.simulate(_request())

    assert exc.value.code == "model_config_missing"

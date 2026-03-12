from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import yaml

from mcp_contracts import (
    ExperimentTrajectory,
    MetadataofRun,
    SimulateEnzymeDynamicsRequest,
    SimulateEnzymeDynamicsResponse,
)
from mcp_errors import ToolExecutionError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

DEFAULT_ENZYME_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMETER_FILE_RELATIVE = Path("config/parameters-123456789.yaml")
DEFAULT_MODEL_CONFIG_RELATIVE = Path("config/config.yaml")
REQUIRED_PARAMETER_NAMES: Tuple[str, ...] = (
    "log_k0_cat",
    "Q10_cat",
    "log_K0_A",
    "Q10_A",
    "log_K0_B",
    "Q10_B",
    "log_K0i_C",
    "Q10_C",
    "log_K0i_D",
    "Q10_D",
    "T_melting",
    "delta_H",
    "delta_C",
)


def _trim(text: str, limit: int = 2000) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


def _read_model_version(enzyme_root: Path) -> str:
    pyproject_path = enzyme_root / "pyproject.toml"
    if not pyproject_path.exists():
        return "unknown"

    try:
        with pyproject_path.open("rb") as stream:
            data = tomllib.load(stream)
        return str(data["project"]["version"])
    except Exception:
        return "unknown"


class EnzymeCliRunner:
    def __init__(
        self,
        enzyme_root: Path = DEFAULT_ENZYME_ROOT,
        python_executable: str = sys.executable,
        timeout_seconds: int = 120,
        parameter_file_relative: Path = DEFAULT_PARAMETER_FILE_RELATIVE,
        model_config_relative: Path = DEFAULT_MODEL_CONFIG_RELATIVE,
    ) -> None:
        self.enzyme_root = enzyme_root
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds
        self.parameter_file_relative = Path(parameter_file_relative)
        self.model_config_relative = Path(model_config_relative)
        self.model_version = _read_model_version(enzyme_root)

    def _validate_static_parameter_file(self, parameters_path: Path) -> None:
        if not parameters_path.exists():
            raise ToolExecutionError(
                code="parameter_file_missing",
                message="Static parameter file does not exist.",
                details={"parameter_file": str(parameters_path)},
            )

        try:
            with parameters_path.open("r", encoding="utf-8") as stream:
                raw_parameters = yaml.safe_load(stream)
        except Exception as exc:
            raise ToolExecutionError(
                code="parameter_file_invalid",
                message="Static parameter file could not be parsed.",
                details={"parameter_file": str(parameters_path), "reason": str(exc)},
            ) from exc

        if not isinstance(raw_parameters, dict):
            raise ToolExecutionError(
                code="parameter_file_invalid",
                message="Static parameter file must contain a mapping of parameter names to values.",
                details={
                    "parameter_file": str(parameters_path),
                    "type": type(raw_parameters).__name__,
                },
            )

        # check that file with params includes all params
        missing = [name for name in REQUIRED_PARAMETER_NAMES if name not in raw_parameters]
        if missing:
            raise ToolExecutionError(
                code="parameter_file_invalid",
                message="Static parameter file is missing required parameters.",
                details={"parameter_file": str(parameters_path), "missing": missing},
            )

        # check that file with params is not corrupted
        for name in REQUIRED_PARAMETER_NAMES:
            try:
                numeric = float(raw_parameters[name])
            except (TypeError, ValueError) as exc:
                raise ToolExecutionError(
                    code="parameter_file_invalid",
                    message="Static parameter file contains non-numeric parameter values.",
                    details={
                        "parameter_file": str(parameters_path),
                        "parameter": name,
                        "reason": str(exc),
                    },
                ) from exc
            if not math.isfinite(numeric):
                raise ToolExecutionError(
                    code="parameter_file_invalid",
                    message="Static parameter file contains non-finite parameter values.",
                    details={
                        "parameter_file": str(parameters_path),
                        "parameter": name,
                        "value": raw_parameters[name],
                    },
                )

    def _read_model_noise_std(self) -> float:
        config_path = self.enzyme_root / self.model_config_relative
        if not config_path.exists():
            raise ToolExecutionError(
                code="model_config_missing",
                message="Model config file does not exist.",
                details={"model_config": str(config_path)},
            )

        try:
            with config_path.open("r", encoding="utf-8") as stream:
                payload = yaml.safe_load(stream)
        except Exception as exc:
            raise ToolExecutionError(
                code="model_config_invalid",
                message="Model config file could not be parsed.",
                details={"model_config": str(config_path), "reason": str(exc)},
            ) from exc

        if not isinstance(payload, dict) or "noise" not in payload:
            raise ToolExecutionError(
                code="model_config_invalid",
                message="Model config file must define a top-level 'noise' value.",
                details={"model_config": str(config_path)},
            )

        try:
            noise_std = float(payload["noise"])
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError(
                code="model_config_invalid",
                message="Model config 'noise' must be numeric.",
                details={"model_config": str(config_path), "value": payload["noise"]},
            ) from exc

        if not math.isfinite(noise_std) or noise_std < 0.0:
            raise ToolExecutionError(
                code="model_config_invalid",
                message="Model config 'noise' must be a finite non-negative number.",
                details={"model_config": str(config_path), "value": payload["noise"]},
            )
        return noise_std

    def _run_command(self, command: list[str]) -> None:
        if not self.enzyme_root.exists():
            raise ToolExecutionError(
                code="enzyme_directory_missing",
                message=f"Enzyme directory does not exist: {self.enzyme_root}",
            )

        try:
            completed = subprocess.run(
                command,
                cwd=self.enzyme_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise ToolExecutionError(
                code="cli_not_found",
                message="Python executable or CLI script was not found.",
                details={"command": command},
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolExecutionError(
                code="cli_timeout",
                message="Enzyme CLI command timed out.",
                details={"command": command, "timeout_seconds": self.timeout_seconds},
            ) from exc

        if completed.returncode != 0:
            raise ToolExecutionError(
                code="cli_execution_failed",
                message="Enzyme CLI returned a non-zero exit status.",
                details={
                    "command": command,
                    "exit_code": completed.returncode,
                    "stdout": _trim(completed.stdout),
                    "stderr": _trim(completed.stderr),
                },
            )

    def simulate(
        self,
        request: SimulateEnzymeDynamicsRequest,
    ) -> SimulateEnzymeDynamicsResponse:
        parameters_path = self.enzyme_root / self.parameter_file_relative
        self._validate_static_parameter_file(parameters_path)
        model_noise_std = self._read_model_noise_std()

        with tempfile.TemporaryDirectory(prefix="enzyme-mcp-") as temp_dir:
            work = Path(temp_dir)
            conditions_path = work / "conditions.json"
            config_path = work / "config.yaml"
            output_path = work / "measurements.json"

            raw_conditions = {
                label: condition.dict()
                for label, condition in request.conditions.items()
            }
            with conditions_path.open("w", encoding="utf-8") as stream:
                json.dump(raw_conditions, stream, indent=2, sort_keys=True)

            config = {
                "experiment": {
                    "duration": float(request.time.t_end - request.time.t_start),
                    # The wrapped CLI drops first/last points, so add one to preserve
                    # the public contract where `measurements` is the number returned.
                    "measurements": int(request.time.measurements + 1),
                },
                "solutions": request.solutions.dict(),
                "noise": model_noise_std,
            }
            with config_path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, sort_keys=True)

            command = [
                self.python_executable,
                "scripts/experiment.py",
                "--parameters",
                str(parameters_path),
                "--conditions",
                str(conditions_path),
                "--output",
                str(output_path),
                "--config",
                str(config_path),
                "--device",
                request.device,
            ]
            self._run_command(command)

            if not output_path.exists():
                raise ToolExecutionError(
                    code="missing_output",
                    message="Simulation completed but did not produce an output file.",
                    details={"output_path": str(output_path)},
                )

            with output_path.open("r", encoding="utf-8") as stream:
                raw_results = json.load(stream)

        if not isinstance(raw_results, dict):
            raise ToolExecutionError(
                code="invalid_cli_output",
                message="Simulation output JSON must be an object keyed by experiment label.",
                details={"type": type(raw_results).__name__},
            )

        experiments: Dict[str, ExperimentTrajectory] = {}
        expected_points = int(request.time.measurements)
        for label, payload in raw_results.items():
            if not isinstance(payload, dict):
                raise ToolExecutionError(
                    code="invalid_cli_output",
                    message="Simulation output payload per experiment must be an object.",
                    details={"label": str(label), "type": type(payload).__name__},
                )

            timestamps_raw = payload.get("timestamps")
            measurements_raw = payload.get("measurements")
            if not isinstance(timestamps_raw, list) or not isinstance(
                measurements_raw, list
            ):
                raise ToolExecutionError(
                    code="invalid_cli_output",
                    message="Simulation output is missing required list fields.",
                    details={
                        "label": str(label),
                        "required_fields": ["timestamps", "measurements"],
                    },
                )

            try:
                timestamps = [
                    float(value) + request.time.t_start for value in timestamps_raw
                ]
                measurements = [float(value) for value in measurements_raw]
            except (TypeError, ValueError) as exc:
                raise ToolExecutionError(
                    code="invalid_cli_output",
                    message="Simulation output contains non-numeric timestamp/measurement values.",
                    details={"label": str(label), "reason": str(exc)},
                ) from exc

            if len(timestamps) != len(measurements):
                raise ToolExecutionError(
                    code="invalid_cli_output",
                    message="Simulation output has mismatched timestamp and measurement lengths.",
                    details={
                        "label": str(label),
                        "timestamps": len(timestamps),
                        "measurements": len(measurements),
                    },
                )
            if len(timestamps) != expected_points:
                raise ToolExecutionError(
                    code="invalid_cli_output",
                    message="Simulation output length does not match requested measurements.",
                    details={
                        "label": str(label),
                        "expected_measurements": expected_points,
                        "actual_measurements": len(timestamps),
                    },
                )

            try:
                experiments[str(label)] = ExperimentTrajectory(
                    time_points=timestamps,
                    state_trajectories={"A_measured": measurements},
                )
            except Exception as exc:
                raise ToolExecutionError(
                    code="invalid_cli_output",
                    message="Simulation output could not be converted into response trajectories.",
                    details={"label": str(label), "reason": str(exc)},
                ) from exc

        warnings: list[str] = []
        if request.time.t_start != 0.0:
            warnings.append(
                "Underlying CLI integrates from t=0; returned timestamps were shifted by t_start."
            )
        if model_noise_std > 0.0:
            warnings.append(
                "Noise is configured by model config and run without an explicit RNG seed."
            )

        # get metadata
        metadata = MetadataofRun(
            model_identifier="enzyme",
            model_version=self.model_version,
            solver={
                "id": "scipy.solve_ivp.LSODA",
                "configuration": {
                    "method": "LSODA",
                    "device": request.device,
                },
            },
            units_map=request.units.dict(),
            warnings=warnings,
            diagnostics={
                "transport": "cli",
                "script": "scripts/experiment.py",
                "enzyme_root": str(self.enzyme_root),
                "parameter_file": str(parameters_path),
                "model_config": str(self.enzyme_root / self.model_config_relative),
                "model_noise_std": model_noise_std,
            },
            deterministic=model_noise_std == 0.0,
            seed=None,
        )

        return SimulateEnzymeDynamicsResponse(
            contract_version=request.contract_version,
            experiments=experiments,
            metadata=metadata,
        )

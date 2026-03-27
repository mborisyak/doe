from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

from mcp_contracts import (
    ExperimentTrajectory,
    MetadataofRun,
    SimulateEnzymeDynamicsRequest,
    SimulateEnzymeDynamicsResponse,
)
from mcp_errors import ToolExecutionError

DEFAULT_ENZYME_ROOT = Path(__file__).resolve().parents[1]


def _trim(text: str, limit: int = 2000) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


class EnzymeCliRunner:
    def __init__(
        self,
        enzyme_root: Path = DEFAULT_ENZYME_ROOT,
        python_executable: str = sys.executable,
        timeout_seconds: int = 120,
    ) -> None:
        self.enzyme_root = enzyme_root
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds

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
        with tempfile.TemporaryDirectory(prefix="enzyme-mcp-") as temp_dir:
            work = Path(temp_dir)
            conditions_path = work / "conditions.json"
            output_path = work / "measurements.json"

            raw_conditions = {
                label: condition.dict()
                for label, condition in request.conditions.items()
            }
            with conditions_path.open("w", encoding="utf-8") as stream:
                json.dump(raw_conditions, stream, indent=2, sort_keys=True)

            command = [
                self.python_executable,
                "scripts/experiment.py",
                "--conditions",
                str(conditions_path),
                "--output",
                str(output_path),
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
                timestamps = [float(value) for value in timestamps_raw]
                measurements = [round(float(value), 3) for value in measurements_raw]
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

            try:
                experiments[str(label)] = ExperimentTrajectory(
                    timestamps=timestamps,
                    A=measurements,
                )
            except Exception as exc:
                raise ToolExecutionError(
                    code="invalid_cli_output",
                    message="Simulation output could not be converted into response trajectories.",
                    details={"label": str(label), "reason": str(exc)},
                ) from exc

        metadata = MetadataofRun(
            model_identifier="enzyme",
            model_version="unknown",
            solver={
                "id": "scipy.solve_ivp.LSODA",
                "configuration": {
                    "method": "LSODA",
                },
            },
            units_map={},
            warnings=[],
            diagnostics={
                "transport": "cli",
                "script": "scripts/experiment.py",
                "enzyme_root": str(self.enzyme_root),
            },
            deterministic=False,
            seed=None,
        )

        return SimulateEnzymeDynamicsResponse(experiments=experiments, metadata=metadata)

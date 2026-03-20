from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List

import numpy as np

try:
    from doe.common import CustomODESystem, ODEModel
except ImportError:  # pragma: no cover
    from doe.doe.common import CustomODESystem, ODEModel

from mcp_contracts import (
    Condition,
    EstimateDoeParametersRequest,
    EstimateDoeParametersResponse,
    ProposeDoeExperimentsRequest,
    ProposeDoeExperimentsResponse,
)
from mcp_errors import ToolExecutionError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOE_ROOT = REPO_ROOT / "doe"


def _ensure_finite_values(name: str, values: Iterable[float]) -> None:
    for value in values:
        if not np.isfinite(value):
            raise ToolExecutionError(
                code="numeric_instability",
                message="Non-finite numeric value encountered during computation.",
                details={"field": name},
            )


def _invalid_model_spec(
    message: str,
    details: Dict[str, Any] | None = None,
) -> ToolExecutionError:
    return ToolExecutionError(
        code="invalid_model_spec",
        message=message,
        details=details,
    )


def _script_error_message(stderr: str, fallback: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("error:"):
            message = line[len("error:") :].strip()
            if message:
                return message
    return fallback


class DoeEngine:
    def __init__(
        self,
        doe_root: Path = DEFAULT_DOE_ROOT,
    ) -> None:
        self.doe_root = doe_root
        self._mle_script = self.doe_root / "scripts" / "mle.py"
        self._new_exp_script = self.doe_root / "scripts" / "new_exp.py"

    def _create_model(
        self,
        model_spec: Dict[str, Any],
    ) -> ODEModel[Any]:
        if len(model_spec) == 0:
            raise _invalid_model_spec("model_spec must be a non-empty object.")
        try:
            return CustomODESystem(model_spec)
        except Exception as exc:
            raise _invalid_model_spec(
                "Failed to construct CustomODESystem from model_spec.",
                details={"type": type(exc).__name__},
            ) from exc

    def _python_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        repo_path = str(self.doe_root)
        existing = env.get("PYTHONPATH")
        if existing:
            env["PYTHONPATH"] = os.pathsep.join([repo_path, existing])
        else:
            env["PYTHONPATH"] = repo_path
        return env

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream)

    @staticmethod
    def _read_json(path: Path, failure_message: str) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError as exc:
            raise ToolExecutionError(
                code="execution_failed",
                message=failure_message,
                details={"reason": "missing_output", "path": str(path)},
            ) from exc
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                code="execution_failed",
                message=failure_message,
                details={
                    "reason": "invalid_json_output",
                    "path": str(path),
                    "type": type(exc).__name__,
                },
            ) from exc

        if not isinstance(payload, dict):
            raise ToolExecutionError(
                code="execution_failed",
                message=failure_message,
                details={"reason": "unexpected_output_shape", "path": str(path)},
            )
        return payload

    def _run_script(
        self,
        *,
        script_path: Path,
        arguments: List[str],
        failure_message: str,
    ) -> None:
        if not script_path.is_file():
            raise ToolExecutionError(
                code="execution_failed",
                message=failure_message,
                details={
                    "reason": "script_not_found",
                    "script": str(script_path),
                },
            )

        cmd = [sys.executable, str(script_path), *arguments]

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self.doe_root),
                env=self._python_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except Exception as exc:
            raise ToolExecutionError(
                code="execution_failed",
                message=failure_message,
                details={"type": type(exc).__name__},
            ) from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            details: Dict[str, Any] = {
                "script": str(script_path),
                "returncode": completed.returncode,
            }
            if stderr:
                details["stderr"] = stderr[-4000:]
            if stdout:
                details["stdout"] = stdout[-4000:]

            raise ToolExecutionError(
                code="execution_failed",
                message=_script_error_message(stderr, failure_message),
                details=details,
            )

    @staticmethod
    def _float_list(value: Any, *, field: str) -> List[float]:
        if not isinstance(value, list) or not value:
            raise ToolExecutionError(
                code="execution_failed",
                message=f"Malformed script output: {field} must be a non-empty list.",
            )

        parsed: List[float] = []
        for item in value:
            try:
                parsed.append(float(item))
            except (TypeError, ValueError) as exc:
                raise ToolExecutionError(
                    code="execution_failed",
                    message=f"Malformed script output: {field} contains non-numeric values.",
                ) from exc
        _ensure_finite_values(field, parsed)
        return parsed

    def estimate_parameters(
        self,
        request: EstimateDoeParametersRequest,
    ) -> EstimateDoeParametersResponse:
        self._create_model(request.model_spec)
        expected_parameter_names = tuple(request.model_spec["parameters"].keys())

        ordered_labels = sorted(request.conditions.keys())
        conditions_payload = {
            label: request.conditions[label].dict() for label in ordered_labels
        }
        measurements_payload = {
            label: request.measurements[label].dict() for label in ordered_labels
        }

        with tempfile.TemporaryDirectory(prefix="mcp-doe-estimate-") as tmp:
            tmp_dir = Path(tmp)
            model_path = tmp_dir / "model.json"
            conditions_path = tmp_dir / "conditions.json"
            measurements_path = tmp_dir / "measurements.json"
            output_path = tmp_dir / "result.json"

            self._write_json(model_path, request.model_spec)
            self._write_json(conditions_path, conditions_payload)
            self._write_json(measurements_path, measurements_payload)

            args = [
                "--model",
                str(model_path),
                "--conditions",
                str(conditions_path),
                "--data",
                str(measurements_path),
                "--iterations",
                str(request.optimizer.iterations),
                "--rtol",
                str(request.optimizer.rtol),
                "--dtype",
                request.optimizer.dtype,
            ]

            if request.initial_parameters is not None:
                initial_path = tmp_dir / "initial.json"
                self._write_json(initial_path, {"parameters": request.initial_parameters})
                args.extend(["--initial", str(initial_path)])

            args.extend(["--output", str(output_path)])
            self._run_script(
                script_path=self._mle_script,
                arguments=args,
                failure_message="Failed to estimate model parameters.",
            )

            result = self._read_json(
                output_path,
                failure_message="Failed to estimate model parameters.",
            )

        raw_parameters = result.get("parameters")
        if not isinstance(raw_parameters, dict):
            raise ToolExecutionError(
                code="execution_failed",
                message="Malformed script output: parameters must be an object.",
            )

        expected_parameter_set = set(expected_parameter_names)
        raw_parameter_set = set(raw_parameters.keys())
        if raw_parameter_set != expected_parameter_set:
            missing = sorted(expected_parameter_set - raw_parameter_set)
            extra = sorted(raw_parameter_set - expected_parameter_set)
            details: List[str] = []
            if missing:
                details.append(f"missing keys: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected keys: {', '.join(extra)}")
            raise ToolExecutionError(
                code="execution_failed",
                message=(
                    "Malformed script output: parameters keys mismatch "
                    f"({'; '.join(details)})."
                ),
            )

        try:
            parameters = {
                name: float(raw_parameters[name]) for name in expected_parameter_names
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolExecutionError(
                code="execution_failed",
                message="Malformed script output: parameters are missing or invalid.",
            ) from exc
        _ensure_finite_values("parameters", parameters.values())

        loss_trace = self._float_list(result.get("loss_trace"), field="loss_trace")

        raw_predictions = result.get("predictions")
        if not isinstance(raw_predictions, dict):
            raise ToolExecutionError(
                code="execution_failed",
                message="Malformed script output: predictions must be an object.",
            )

        prediction_payload: Dict[str, List[float]] = {}
        for label in ordered_labels:
            if label not in raw_predictions:
                raise ToolExecutionError(
                    code="execution_failed",
                    message="Malformed script output: missing prediction label.",
                    details={"label": label},
                )

            series = raw_predictions[label]
            if not isinstance(series, list) or not series:
                raise ToolExecutionError(
                    code="numeric_instability",
                    message="Prediction output has an unexpected shape.",
                    details={"label": label},
                )

            try:
                parsed_series = [float(v) for v in series]
            except (TypeError, ValueError) as exc:
                raise ToolExecutionError(
                    code="execution_failed",
                    message="Malformed script output: prediction values must be numeric.",
                    details={"label": label},
                ) from exc
            _ensure_finite_values(f"predictions.{label}", parsed_series)
            prediction_payload[label] = parsed_series

        return EstimateDoeParametersResponse(
            parameters=parameters,
            loss_trace=loss_trace,
            predictions=prediction_payload,
        )

    def propose_experiments(
        self,
        request: ProposeDoeExperimentsRequest,
    ) -> ProposeDoeExperimentsResponse:
        self._create_model(request.model_spec)

        with tempfile.TemporaryDirectory(prefix="mcp-doe-propose-") as tmp:
            tmp_dir = Path(tmp)
            model_path = tmp_dir / "model.json"
            output_path = tmp_dir / "result.json"

            self._write_json(model_path, request.model_spec)

            args = [
                "--model",
                str(model_path),
                "--n",
                str(request.proposal_config.n_proposals),
                "--iterations",
                str(request.proposal_config.iterations),
                "--criterion",
                request.proposal_config.criterion,
                "--seed",
                str(request.proposal_config.seed),
                "--output",
                str(output_path),
            ]

            if request.history is not None:
                ordered_labels = sorted(request.history.conditions.keys())
                history_conditions = {
                    label: request.history.conditions[label].dict()
                    for label in ordered_labels
                }
                history_data = {
                    label: {
                        "timestamps": request.history.timestamps[label],
                        "measurements": request.history.measurements[label],
                    }
                    for label in ordered_labels
                }
                conditions_path = tmp_dir / "conditions.json"
                history_data_path = tmp_dir / "history_data.json"
                self._write_json(conditions_path, history_conditions)
                self._write_json(history_data_path, history_data)
                args.extend([
                    "--conditions", str(conditions_path),
                    "--data", str(history_data_path),
                ])

            if request.parameters is not None:
                parameters_path = tmp_dir / "parameters.json"
                self._write_json(parameters_path, {"parameters": request.parameters})
                args.extend(["--parameters", str(parameters_path)])

            if request.proposal_config.regularization is not None:
                args.extend(
                    [
                        "--regularization",
                        str(request.proposal_config.regularization),
                    ]
                )

            self._run_script(
                script_path=self._new_exp_script,
                arguments=args,
                failure_message="Failed to propose new experiments.",
            )

            result = self._read_json(
                output_path,
                failure_message="Failed to propose new experiments.",
            )

        raw_proposals = result.get("proposals")
        if not isinstance(raw_proposals, list) or not raw_proposals:
            raise ToolExecutionError(
                code="execution_failed",
                message="Malformed script output: proposals must be a non-empty list.",
            )

        proposed_conditions: List[Condition] = []
        for proposal in raw_proposals:
            try:
                condition = Condition.parse_obj(proposal)
            except Exception as exc:
                raise ToolExecutionError(
                    code="execution_failed",
                    message="Malformed script output: invalid proposed condition.",
                    details={"type": type(exc).__name__},
                ) from exc
            _ensure_finite_values("proposed_conditions", condition.dict().values())
            proposed_conditions.append(condition)

        proposal_timestamps = self._float_list(result.get("proposal_timestamps"), field="proposal_timestamps")

        raw_expected = result.get("expected")
        if not isinstance(raw_expected, list) or not raw_expected:
            raise ToolExecutionError(
                code="execution_failed",
                message="Malformed script output: expected must be a non-empty list.",
            )

        expected: List[List[float]] = []
        for idx, row in enumerate(raw_expected):
            if not isinstance(row, list):
                raise ToolExecutionError(
                    code="execution_failed",
                    message="Malformed script output: each expected trajectory must be a list.",
                    details={"index": idx},
                )
            try:
                parsed_row = [float(v) for v in row]
            except (TypeError, ValueError) as exc:
                raise ToolExecutionError(
                    code="execution_failed",
                    message="Malformed script output: expected trajectory values must be numeric.",
                    details={"index": idx},
                ) from exc
            _ensure_finite_values(f"expected[{idx}]", parsed_row)
            expected.append(parsed_row)

        return ProposeDoeExperimentsResponse(
            proposed_conditions=proposed_conditions,
            proposal_timestamps=proposal_timestamps,
            expected=expected,
        )

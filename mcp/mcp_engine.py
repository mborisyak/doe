from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

try:
    from doe.common import CustomODESystem, ODEModel
except ImportError:  # pragma: no cover
    from doe.doe.common import CustomODESystem, ODEModel

from mcp_contracts import (
    EstimateDoeParametersRequest,
    ProposeDoeExperimentsRequest,
)
from mcp_errors import ToolExecutionError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOE_ROOT = REPO_ROOT / "doe"


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
        scripts = self.doe_root / "scripts"
        self._mle_script = scripts / "mle.py"
        self._new_exp_script = scripts / "new_exp.py"
        self._gp_fit_script = scripts / "gp_fit.py"
        self._gp_hyperfit_script = scripts / "gp_hyperfit.py"
        self._gp_predict_script = scripts / "gp_predict.py"
        self._gp_doe_script = scripts / "gp_doe.py"
        self._gp_discriminate_script = scripts / "gp_discriminate.py"

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
    ) -> str:
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

        return completed.stdout or ""

    @classmethod
    def _materialise_batches(cls, tmp_dir: Path, batches: Dict[str, Any]) -> List[str]:
        """Write each ``{batch-name: record}`` to ``<tmp>/<batch-name>.json`` so the script
        sees the batch name as the file stem (see doe.dataset). Returns the paths."""
        paths: List[str] = []
        for name, record in batches.items():
            path = tmp_dir / f"{name}.json"
            cls._write_json(path, record)
            paths.append(str(path))
        return paths

    def estimate_parameters(self, request: EstimateDoeParametersRequest) -> Dict[str, Any]:
        # Materialise the resolved inputs to a temp dir, run the store-free script there, and
        # return its record body verbatim (parameters + fit diagnostics + auxiliary). The MCP
        # decides nothing about the body shape -- compute_tools just stamps linkage refs.
        self._create_model(request.model_spec)

        with tempfile.TemporaryDirectory(prefix="mcp-doe-estimate-") as tmp:
            tmp_dir = Path(tmp)
            model_path = tmp_dir / "model.json"
            output_path = tmp_dir / "result.json"
            self._write_json(model_path, request.model_spec)
            batch_paths = self._materialise_batches(tmp_dir, request.batches)

            args = [
                "--model", str(model_path),
                "--batches", *batch_paths,
                "--iterations", str(request.optimizer.iterations),
                "--rtol", str(request.optimizer.rtol),
                "--dtype", request.optimizer.dtype,
            ]
            if request.initial_parameters is not None:
                initial_path = tmp_dir / "initial.json"
                self._write_json(initial_path, request.initial_parameters)
                args.extend(["--initial", str(initial_path)])
            args.extend(["--output", str(output_path)])

            self._run_script(
                script_path=self._mle_script, arguments=args,
                failure_message="Failed to estimate model parameters.",
            )
            return self._read_json(output_path, failure_message="Failed to estimate model parameters.")

    def propose_experiments(self, request: ProposeDoeExperimentsRequest) -> Dict[str, Any]:
        # As with estimate_parameters: run the script, return its design record body verbatim.
        self._create_model(request.model_spec)

        with tempfile.TemporaryDirectory(prefix="mcp-doe-propose-") as tmp:
            tmp_dir = Path(tmp)
            model_path = tmp_dir / "model.json"
            output_path = tmp_dir / "result.json"
            self._write_json(model_path, request.model_spec)

            args = [
                "--model", str(model_path),
                "--n", str(request.proposal_config.n_proposals),
                "--iterations", str(request.proposal_config.iterations),
                "--criterion", request.proposal_config.criterion,
                "--seed", str(request.proposal_config.seed),
                "--output", str(output_path),
            ]
            if request.parameters is not None:
                params_path = tmp_dir / "parameters.json"
                self._write_json(params_path, request.parameters)
                args.extend(["--parameters", str(params_path)])
            if request.history:
                history_paths = self._materialise_batches(tmp_dir, request.history)
                args.extend(["--history", *history_paths])
            if request.proposal_config.regularization is not None:
                args.extend(["--regularization", str(request.proposal_config.regularization)])

            self._run_script(
                script_path=self._new_exp_script, arguments=args,
                failure_message="Failed to propose new experiments.",
            )
            return self._read_json(output_path, failure_message="Failed to propose new experiments.")

    # --------------------------------------------------------------- GP (subprocess)
    # Each GP tool materialises its resolved inputs into a temp dir, runs the store-free GP
    # script there, and returns the parsed result dict (the resolver commits any record).
    def fit_gp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="mcp-gp-fit-") as tmp:
            tmp_dir = Path(tmp)
            model_path = tmp_dir / "model.json"
            output_path = tmp_dir / "result.json"
            self._write_json(model_path, request["model_spec"])
            batch_paths = self._materialise_batches(tmp_dir, request["batches"])
            args = ["--model", str(model_path), "--batches", *batch_paths, "--output", str(output_path)]
            if request.get("folds") is not None:
                args += ["--folds", str(request["folds"])]
            self._run_script(script_path=self._gp_fit_script, arguments=args, failure_message="Failed to fit GP.")
            return self._read_json(output_path, failure_message="Failed to fit GP.")

    def hyper_fit_gp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="mcp-gp-hyperfit-") as tmp:
            tmp_dir = Path(tmp)
            model_path = tmp_dir / "model.json"
            output_path = tmp_dir / "result.json"
            self._write_json(model_path, request["model_spec"])
            batch_paths = self._materialise_batches(tmp_dir, request["batches"])
            args = ["--model", str(model_path), "--batches", *batch_paths, "--output", str(output_path)]
            self._run_script(script_path=self._gp_hyperfit_script, arguments=args,
                             failure_message="Failed to hyper-fit GP.")
            return self._read_json(output_path, failure_message="Failed to hyper-fit GP.")

    def predict_gp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="mcp-gp-predict-") as tmp:
            tmp_dir = Path(tmp)
            paths = {n: tmp_dir / f"{n}.json" for n in ("model", "state", "points")}
            output_path = tmp_dir / "result.json"
            self._write_json(paths["model"], request["model_spec"])
            self._write_json(paths["state"], request["state"])
            self._write_json(paths["points"], request["points"])
            args = ["--model", str(paths["model"]), "--state", str(paths["state"]),
                    "--points", str(paths["points"]), "--output", str(output_path)]
            self._run_script(script_path=self._gp_predict_script, arguments=args,
                             failure_message="Failed to predict with GP.")
            return self._read_json(output_path, failure_message="Failed to predict with GP.")

    def doe_gp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="mcp-gp-doe-") as tmp:
            tmp_dir = Path(tmp)
            paths = {n: tmp_dir / f"{n}.json" for n in ("model", "state", "bounds")}
            output_path = tmp_dir / "result.json"
            self._write_json(paths["model"], request["model_spec"])
            self._write_json(paths["state"], request["state"])
            self._write_json(paths["bounds"], request["bounds"])
            args = ["--model", str(paths["model"]), "--state", str(paths["state"]),
                    "--bounds", str(paths["bounds"]), "--batch-size", str(request["batch_size"]),
                    "--seed", str(request.get("seed", 0)), "--output", str(output_path)]
            self._run_script(script_path=self._gp_doe_script, arguments=args,
                             failure_message="Failed to run GP DoE.")
            return self._read_json(output_path, failure_message="Failed to run GP DoE.")

    def discriminate_gp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="mcp-gp-discriminate-") as tmp:
            tmp_dir = Path(tmp)
            paths = {n: tmp_dir / f"{n}.json" for n in ("model", "state", "grid", "bounds")}
            output_path = tmp_dir / "result.json"
            self._write_json(paths["model"], request["model_spec"])
            self._write_json(paths["state"], request["state"])
            self._write_json(paths["grid"], request["grid"])
            self._write_json(paths["bounds"], request["bounds"])
            args = ["--model", str(paths["model"]), "--state", str(paths["state"]),
                    "--grid", str(paths["grid"]), "--threshold", str(request["threshold"]),
                    "--bounds", str(paths["bounds"]), "--batch-size", str(request["batch_size"]),
                    "--seed", str(request.get("seed", 0)), "--output", str(output_path)]
            self._run_script(script_path=self._gp_discriminate_script, arguments=args,
                             failure_message="Failed to run discriminative GP DoE.")
            return self._read_json(output_path, failure_message="Failed to run discriminative GP DoE.")

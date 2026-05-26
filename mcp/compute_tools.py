"""By-name compute resolver (see STORAGE.md).

The MCP compute tools take store names, not copied numbers. For each call this
layer resolves + verifies the input refs into the inline payload the pure core
expects, **reserves** the output name (``group.lock``), runs the (unchanged,
path-based) core via the existing inline services, writes the result record + a
``tool_result``, and returns the ``references``. Name resolution and verification
live here; the scripts stay store-agnostic and take arbitrary file paths.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from doe.store import Store, StoreError, view
from mcp_tools import DoeMcpService, EnzymeMcpService

_CALL_PREFIX = {
    "fit_parameters": "fit",
    "propose_doe_experiments": "propose",
    "simulate_enzyme_dynamics": "simulate",
}


class _Abort(Exception):
    """Raised inside a reservation to roll it back (free the name) and surface a
    compute-error envelope unchanged."""

    def __init__(self, envelope: Dict[str, Any]) -> None:
        self.envelope = envelope


def _error(exc: StoreError) -> Dict[str, Any]:
    return {"ok": False, "error": {"code": exc.code, "message": exc.message, "details": exc.details}}


class ComputeResolver:
    def __init__(
        self,
        store: Store,
        doe_service: Optional[DoeMcpService] = None,
        enzyme_service: Optional[EnzymeMcpService] = None,
    ) -> None:
        self.store = store
        self.doe = doe_service or DoeMcpService()
        self.enzyme = enzyme_service or EnzymeMcpService()

    # ------------------------------------------------------------ ref helpers
    def _typed(self, ref: str, expected: str) -> Dict[str, Any]:
        record = self.store.read(ref)  # raises StoreError(not_found)
        if record.get("type") != expected:
            raise StoreError(
                "wrong_type",
                f"{ref!r} is a {record.get('type')!r}, expected {expected!r}",
                {"ref": ref, "expected": expected},
            )
        return record

    def _model_spec(self, model: str) -> Dict[str, Any]:
        return self._typed(model, "model")["spec"]

    def _fitted(self, fitted_model: str) -> Tuple[Dict[str, Any], Dict[str, float], str]:
        record = self._typed(fitted_model, "fitted_model")
        spec = self._typed(record["model"], "model")["spec"]
        return spec, record["parameters"], record["model"]

    def _batch_cm(self, batches: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        conditions: Dict[str, Any] = {}
        measurements: Dict[str, Any] = {}
        for batch in batches:
            record = self._typed(batch, "batch")
            aux = record.get("auxiliary", {})
            for exp, payload in record["experiments"].items():
                label = f"{batch}/{exp}"
                series = aux.get(exp, {})
                conditions[label] = payload["x"]
                measurements[label] = {"timestamps": series["t"], "measurements": series["y"]}
        return conditions, measurements

    def _batch_history(self, batches: Sequence[str]) -> Dict[str, Any]:
        conditions: Dict[str, Any] = {}
        timestamps: Dict[str, Any] = {}
        measurements: Dict[str, Any] = {}
        for batch in batches:
            record = self._typed(batch, "batch")
            aux = record.get("auxiliary", {})
            for exp, payload in record["experiments"].items():
                label = f"{batch}/{exp}"
                series = aux.get(exp, {})
                conditions[label] = payload["x"]
                timestamps[label] = series["t"]
                measurements[label] = series["y"]
        return {"conditions": conditions, "timestamps": timestamps, "measurements": measurements}

    def _design_conditions(self, design: str) -> Dict[str, Any]:
        record = self._typed(design, "design")
        return {exp: payload["x"] for exp, payload in record["experiments"].items()}

    @staticmethod
    def _observable(spec: Dict[str, Any]) -> str:
        observables = spec.get("observables")
        if isinstance(observables, dict) and observables:
            return next(iter(observables))
        return "observable"

    @staticmethod
    def _call_name(call: str, tool: str, output: str) -> str:
        if call and call != "-":
            return call
        return f"{_CALL_PREFIX[tool]}/{output}"

    def _write_tool_result(
        self, call_name: str, tool: str, inputs: Dict[str, Any], references: Dict[str, Any], envelope_data: Any
    ) -> None:
        results = self.store["tool_result"]
        with results.lock(call_name):
            results[call_name] = {
                "tool": tool,
                "status": "ok",
                "inputs": {k: v for k, v in inputs.items() if v is not None},
                "references": references,
                "auxiliary": {"envelope": envelope_data},
            }

    # ------------------------------------------------------------ tools
    def fit(
        self,
        *,
        model: str,
        data,
        name: str = "-",
        initial: Optional[str] = None,
        call: str = "-",
        optimizer: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        batches: List[str] = [data] if isinstance(data, str) else list(data)
        try:
            conditions, measurements = self._batch_cm(batches)
            request: Dict[str, Any] = {
                "model_spec": self._model_spec(model),
                "conditions": conditions,
                "measurements": measurements,
            }
            if initial is not None:
                _, params, _ = self._fitted(initial)
                request["initial_parameters"] = params
            if optimizer:
                request["optimizer"] = optimizer
        except StoreError as exc:
            return _error(exc)

        fits = self.store["fitted_model"][model]
        try:
            with fits.lock(name) as version:
                envelope = self.doe.fit_parameters(request)
                if not envelope.get("ok"):
                    raise _Abort(envelope)
                payload = envelope["data"]
                full = f"{model}/{version}"
                call_name = self._call_name(call, "fit_parameters", full)
                fits[version] = {
                    "description": description,
                    "model": model,
                    "parameters": payload["parameters"],
                    "fit": {"data": batches, "tool_result": call_name, "final_loss": payload["loss_trace"][-1]},
                    "auxiliary": {"loss_trace": payload["loss_trace"], "predictions": payload["predictions"]},
                }
        except _Abort as abort:
            return abort.envelope
        except StoreError as exc:
            return _error(exc)

        references = {"call": call_name, "fitted_model": full, "model": model, "data": batches}
        if initial is not None:
            references["initial"] = initial
        self._write_tool_result(
            call_name, "fit_parameters",
            {"model": model, "data": batches, "initial": initial, "optimizer": optimizer},
            references, payload,
        )
        return {"ok": True, "references": references, "data": view(self.store.read(full))}

    def propose(
        self,
        *,
        fitted_model: str,
        proposal_config: Dict[str, Any],
        name: str = "-",
        history=None,
        call: str = "-",
        description: str = "",
    ) -> Dict[str, Any]:
        hist: Optional[List[str]] = None
        try:
            spec, params, _ = self._fitted(fitted_model)
            request: Dict[str, Any] = {"model_spec": spec, "parameters": params, "proposal_config": proposal_config}
            if history:
                hist = [history] if isinstance(history, str) else list(history)
                request["history"] = self._batch_history(hist)
        except StoreError as exc:
            return _error(exc)

        designs = self.store["design"]
        try:
            with designs.lock(name) as design:
                envelope = self.doe.propose_doe_experiments(request)
                if not envelope.get("ok"):
                    raise _Abort(envelope)
                payload = envelope["data"]
                call_name = self._call_name(call, "propose_doe_experiments", design)
                conditions = payload["proposed_conditions"]
                timestamps = payload["proposal_timestamps"]
                expected = payload["expected"]
                experiments, aux, expected_by_exp = {}, {}, {}
                for i, condition in enumerate(conditions):
                    label = f"experiment-{i + 1}"
                    experiments[label] = {"x": condition}
                    aux[label] = {"t": timestamps}
                    expected_by_exp[label] = expected[i]
                designs[design] = {
                    "description": description,
                    "space": list(conditions[0].keys()) if conditions else [],
                    "observable": self._observable(spec),
                    "experiments": experiments,
                    "auxiliary": {**aux, "expected": {fitted_model: expected_by_exp}},
                    "source": {"tool_result": call_name, "model": fitted_model},
                }
        except _Abort as abort:
            return abort.envelope
        except StoreError as exc:
            return _error(exc)

        references = {"call": call_name, "design": design, "fitted_model": fitted_model}
        if hist:
            references["history"] = hist
        self._write_tool_result(
            call_name, "propose_doe_experiments",
            {"fitted_model": fitted_model, "history": hist, "proposal_config": proposal_config},
            references, payload,
        )
        return {"ok": True, "references": references, "data": view(self.store.read(design))}

    def simulate(
        self,
        *,
        name: str = "-",
        design: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
        call: str = "-",
        time: Optional[Dict[str, Any]] = None,
        solutions: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        try:
            if design is not None:
                conds = self._design_conditions(design)
            elif conditions is not None:
                conds = conditions
            else:
                return {"ok": False, "error": {"code": "invalid_request",
                                               "message": "simulate_enzyme_dynamics needs `design` or `conditions`."}}
            request: Dict[str, Any] = {"conditions": conds}
            if time:
                request["time"] = time
            if solutions:
                request["solutions"] = solutions
            if device:
                request["device"] = device
        except StoreError as exc:
            return _error(exc)

        batches = self.store["batch"]
        try:
            with batches.lock(name) as batch:
                envelope = self.enzyme.simulate_enzyme_dynamics(request)
                if not envelope.get("ok"):
                    raise _Abort(envelope)
                payload = envelope["data"]
                call_name = self._call_name(call, "simulate_enzyme_dynamics", batch)
                experiments, aux = {}, {}
                for label, traj in payload["experiments"].items():
                    experiments[label] = {"x": conds[label]}
                    aux[label] = {"t": traj["time_points"], "y": traj["state_trajectories"]["A_measured"]}
                batches[batch] = {
                    "description": description,
                    "space": ["A", "B", "E", "temperature"],
                    "observable": "A_measured",
                    "design": design,
                    "experiments": experiments,
                    "auxiliary": aux,
                    "source": {"tool_result": call_name},
                }
        except _Abort as abort:
            return abort.envelope
        except StoreError as exc:
            return _error(exc)

        references = {"call": call_name, "batch": batch}
        if design is not None:
            references["design"] = design
        self._write_tool_result(
            call_name, "simulate_enzyme_dynamics",
            {"design": design, "conditions": None if design else conds,
             "time": time, "solutions": solutions, "device": device},
            references, {"metadata": payload.get("metadata")},
        )
        return {"ok": True, "references": references, "data": view(self.store.read(batch))}

"""By-name compute resolver (analysis/design: fit / propose / GP).

The MCP compute tools take store names, not copied numbers. For each call this layer
resolves + verifies the input refs, **reserves** the output name (``group.lock``), runs
the (store-agnostic, path-based) script as a subprocess, then stamps + commits the result
record + a ``logs`` entry and returns only the ``references`` (no record dump). Shared
store helpers come from :class:`StoreResolver`. Simulation (the data source) lives
separately in ``simulate_tools.py``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from doe.store import Store, StoreError
from mcp_tools import DoeMcpService
from resolver_base import StoreResolver, _Abort, _error

# GP input layout (conditions then time); kept local so this module stays JAX-free -- all
# GP numerics run in the subprocess scripts.
_GP_INPUT_VARS = ("A", "B", "E", "temperature")
_GP_BOUND_VARS = _GP_INPUT_VARS + ("t",)


class ComputeResolver(StoreResolver):
    def __init__(self, store: Store, doe_service: Optional[DoeMcpService] = None) -> None:
        super().__init__(store)
        self.doe = doe_service or DoeMcpService()

    # ------------------------------------------------------------ ref helpers
    def _model_spec(self, model: str) -> Dict[str, Any]:
        return self._typed(model, "model")["spec"]

    def _fitted(self, fitted_model: str) -> Tuple[Dict[str, Any], Dict[str, float], str]:
        record = self._typed(fitted_model, "fitted_model")
        spec = self._typed(record["model"], "model")["spec"]
        return spec, record["parameters"], record["model"]

    def _root(self) -> str:
        """Absolute store root, so the script's subprocess opens the same store
        regardless of its working directory."""
        return str(self.store.root.resolve())

    # ------------------------------------------------------------ payload stamps
    # The compute scripts own the record *body* (parameters/experiments + fit + auxiliary).
    # These helpers dump that body and add only the linkage refs the store layer needs --
    # the MCP decides nothing about the body shape.
    @staticmethod
    def _fitted_record(body: Dict[str, Any], *, model: str, batches: List[str],
                       call_name: str, description: str) -> Dict[str, Any]:
        record = dict(body)
        record["description"] = description
        record["model"] = model
        record["fit"] = {**record.get("fit", {}), "data": batches, "tool_result": call_name}
        return record

    @staticmethod
    def _design_record(body: Dict[str, Any], *, fitted_model: str, call_name: str,
                       description: str, ref_key: str = "fitted_model") -> Dict[str, Any]:
        record = dict(body)
        record["description"] = description
        record["source"] = {"tool_result": call_name, ref_key: fitted_model}
        # auxiliary.expected is the model's expected output; key it by the fitted_model ref.
        aux = dict(record.get("auxiliary") or {})
        if "expected" in aux:
            aux["expected"] = {fitted_model: aux["expected"]}
        record["auxiliary"] = aux
        return record

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
            # resolve refs -> the spec + batch records the script runs on
            request: Dict[str, Any] = {
                "model_spec": self._model_spec(model),
                "batches": {b: self._typed(b, "data") for b in batches},
            }
            if initial is not None:
                _, params, _ = self._fitted(initial)
                request["initial_parameters"] = params
            if optimizer:
                request["optimizer"] = optimizer
        except StoreError as exc:
            return _error(exc)

        try:
            # Reserve BOTH names before the (long) compute -- the output (minted vN under the
            # model) and the call (fit_parameters/<model>-<N>) -- so a duplicate fails here,
            # not after the run completes.
            with self.store["fitted_model"][model].lock(name) as version, \
                    self.store["logs"]["fit_ode"].lock(call, f"{model}-") as leaf:
                full, call_name = f"{model}/{version}", f"fit_ode/{leaf}"
                envelope = self.doe.fit_parameters(request)
                if not envelope.get("ok"):
                    raise _Abort(envelope)
                outputs = {"fitted_model": full}
                # Store._write stamps type/name/created_at; the script owns the rest of the body.
                # Write through the proper nested path (the reserved groups), not a compound key.
                self.store["fitted_model"][model][version] = self._fitted_record(
                    envelope["data"], model=model, batches=batches,
                    call_name=call_name, description=description)
                self.store["logs"]["fit_ode"][leaf] = self._tool_result(
                    "fit_ode",
                    {"optimizer": optimizer},                                # arguments (non-ref)
                    {"model": model, "data": batches, "initial": initial},   # inputs (refs)
                    outputs,                                                 # outputs (produced refs)
                )
        except _Abort as abort:
            return abort.envelope
        except StoreError as exc:
            return _error(exc)
        return {"ok": True, "references": {"call": call_name, **outputs}}

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
                request["history"] = {b: self._typed(b, "data") for b in hist}
        except StoreError as exc:
            return _error(exc)

        try:
            # Reserve the design + the call (propose_doe_experiments/<fitted_model>-<N>) up front.
            with self.store["design"].lock(name) as design, \
                    self.store["logs"]["doe_ode"].lock(call, f"{fitted_model}-") as leaf:
                call_name = f"doe_ode/{leaf}"
                envelope = self.doe.propose_doe_experiments(request)
                if not envelope.get("ok"):
                    raise _Abort(envelope)
                outputs = {"design": design}
                # The script returns the design body (experiments + auxiliary.expected keyed by
                # observable); stamp the source ref and re-key expected by the fitted_model.
                self.store["design"][design] = self._design_record(
                    envelope["data"], fitted_model=fitted_model, call_name=call_name,
                    description=description, ref_key="model")
                self.store["logs"]["doe_ode"][leaf] = self._tool_result(
                    "doe_ode",
                    {"proposal_config": proposal_config},              # arguments (non-ref)
                    {"fitted_model": fitted_model, "history": hist},   # inputs (refs)
                    outputs,                                           # outputs (produced refs)
                )
        except _Abort as abort:
            return abort.envelope
        except StoreError as exc:
            return _error(exc)
        return {"ok": True, "references": {"call": call_name, **outputs}}

    # --------------------------------------------------------------- GP (by name)
    # The resolver stays JAX-free: it resolves refs, the engine runs the GP scripts in a
    # subprocess, and the resolver commits the records.
    @staticmethod
    def _gp_spec(kernel, length_scales, length_scale, variance, noise, observables) -> Dict[str, Any]:
        spec: Dict[str, Any] = {
            "kernel": kernel,
            "variance": float(variance),
            "noise": float(noise),
            "observables": list(observables) if observables else ["A"],
        }
        if length_scales is not None:
            spec["length_scales"] = [float(v) for v in length_scales]
        elif length_scale is not None:
            spec["length_scale"] = float(length_scale)
        return spec

    @staticmethod
    def _gp_bounds(bounds: Dict[str, Any]) -> List[List[float]]:
        """Order the agent's ``{var: [lo, hi]}`` over the GP input dims (A, B, E, temperature, t)."""
        return [[float(bounds[v][0]), float(bounds[v][1])] for v in _GP_BOUND_VARS]

    def _gp_fitted(self, fitted_model: str):
        """Resolve a GP fitted_model -> (model_spec, gp_state); state is None if absent."""
        record = self._typed(fitted_model, "fitted_model")
        spec = self._typed(record["model"], "model")["spec"]
        return spec, (record.get("auxiliary") or {}).get("gp_state")

    def create_gp(
        self,
        *,
        name: str = "-",
        kernel: str = "rbf",
        length_scales: Optional[Sequence[float]] = None,
        length_scale: Optional[float] = None,
        variance: float = 1.0,
        noise: float = 1.0e-6,
        observables: Optional[Sequence[str]] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        # No compute -- just assemble the spec and register the model.
        spec = self._gp_spec(kernel, length_scales, length_scale, variance, noise, observables)
        try:
            with self.store["model"].lock(name, "gp-") as model:  # GP models mint as gp-1, gp-2, …
                self.store["model"][model] = {"kind": "gp", "spec": spec, "description": description}
        except StoreError as exc:
            return _error(exc)
        return {"ok": True, "references": {"model": model}}

    def hyper_fit_gp(
        self,
        *,
        data,
        name: str = "-",
        kernel: str = "rbf",
        noise: float = 1.0e-6,
        observables: Optional[Sequence[str]] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        batches: List[str] = [data] if isinstance(data, str) else list(data)
        base = {"kernel": kernel, "noise": float(noise), "observables": list(observables) if observables else ["A"]}
        try:
            request = {"model_spec": base, "batches": {b: self._typed(b, "data") for b in batches}}
        except StoreError as exc:
            return _error(exc)
        try:
            with self.store["model"].lock(name, "gp-") as model:  # GP models mint as gp-1, gp-2, …
                envelope = self.doe.hyper_fit_gp(request)
                if not envelope.get("ok"):
                    raise _Abort(envelope)
                self.store["model"][model] = {"kind": "gp", "spec": envelope["data"]["spec"],
                                              "description": description, "source": {"data": batches}}
        except _Abort as abort:
            return abort.envelope
        except StoreError as exc:
            return _error(exc)
        return {"ok": True, "references": {"model": model, "data": batches}}

    def fit_gp(
        self,
        *,
        model: str,
        data,
        name: str = "-",
        call: str = "-",
        folds: int = 5,
        description: str = "",
    ) -> Dict[str, Any]:
        batches: List[str] = [data] if isinstance(data, str) else list(data)
        try:
            request = {
                "model_spec": self._typed(model, "model")["spec"],
                "batches": {b: self._typed(b, "data") for b in batches},
                "folds": folds,
            }
        except StoreError as exc:
            return _error(exc)

        try:
            with self.store["fitted_model"][model].lock(name) as version, \
                    self.store["logs"]["fit_gp"].lock(call, f"{model}-") as leaf:
                full, call_name = f"{model}/{version}", f"fit_gp/{leaf}"
                envelope = self.doe.fit_gp(request)
                if not envelope.get("ok"):
                    raise _Abort(envelope)
                outputs = {"fitted_model": full}
                self.store["fitted_model"][model][version] = self._fitted_record(
                    envelope["data"], model=model, batches=batches,
                    call_name=call_name, description=description)
                self.store["logs"]["fit_gp"][leaf] = self._tool_result(
                    "fit_gp", {"folds": folds}, {"model": model, "data": batches}, outputs)
        except _Abort as abort:
            return abort.envelope
        except StoreError as exc:
            return _error(exc)
        return {"ok": True, "references": {"call": call_name, **outputs}}

    def predict_gp(self, *, fitted_model: str, points: Sequence[Sequence[float]]) -> Dict[str, Any]:
        try:
            spec, state = self._gp_fitted(fitted_model)
        except StoreError as exc:
            return _error(exc)
        if state is None:
            return {"ok": False, "error": {"code": "invalid_request",
                                           "message": f"{fitted_model!r} has no gp_state (not a GP fit?)"}}
        envelope = self.doe.predict_gp({"model_spec": spec, "state": state, "points": list(points)})
        if not envelope.get("ok"):
            return envelope
        return {"ok": True, "references": {"fitted_model": fitted_model}, "data": envelope["data"]}

    def _gp_design(self, *, fitted_model, name, call, tool, description, arguments, run):
        """Reserve the design name + agent-supplied call_name, run the (long) GP DoE compute
        ``run()`` with both held, then commit the design body it returns (experiments +
        auxiliary.expected + eig). Names are locked before the compute, not after."""
        try:
            with self.store["design"].lock(name) as design, \
                    self.store["logs"][tool].lock(call, f"{fitted_model}-") as leaf:
                call_name = f"{tool}/{leaf}"
                envelope = run()
                if not envelope.get("ok"):
                    raise _Abort(envelope)
                outputs = {"design": design}
                self.store["design"][design] = self._design_record(
                    envelope["data"], fitted_model=fitted_model, call_name=call_name,
                    description=description)
                self.store["logs"][tool][leaf] = self._tool_result(
                    tool, arguments, {"fitted_model": fitted_model}, outputs)
        except _Abort as abort:
            return abort.envelope
        except StoreError as exc:
            return _error(exc)
        return {"ok": True, "references": {"call": call_name, **outputs}}

    def doe_gp(
        self,
        *,
        fitted_model: str,
        bounds: Dict[str, Any],
        batch_size: int,
        name: str = "-",
        call: str = "-",
        seed: int = 0,
        description: str = "",
    ) -> Dict[str, Any]:
        try:
            spec, state = self._gp_fitted(fitted_model)
            ordered = self._gp_bounds(bounds)
        except StoreError as exc:
            return _error(exc)
        except (KeyError, TypeError) as exc:
            return {"ok": False, "error": {"code": "invalid_request",
                                           "message": f"bounds must map each of {list(_GP_BOUND_VARS)} to [lo, hi] ({exc})"}}
        if state is None:
            return {"ok": False, "error": {"code": "invalid_request", "message": f"{fitted_model!r} has no gp_state"}}
        return self._gp_design(
            fitted_model=fitted_model, name=name, call=call, tool="doe_gp", description=description,
            arguments={"bounds": bounds, "batch_size": batch_size, "seed": seed},
            run=lambda: self.doe.doe_gp({"model_spec": spec, "state": state, "bounds": ordered,
                                         "batch_size": batch_size, "seed": seed}),
        )

    def discriminatory_doe_gp(
        self,
        *,
        fitted_model: str,
        grid: Sequence[Sequence[float]],
        threshold: float,
        bounds: Dict[str, Any],
        batch_size: int,
        name: str = "-",
        call: str = "-",
        seed: int = 0,
        description: str = "",
    ) -> Dict[str, Any]:
        try:
            spec, state = self._gp_fitted(fitted_model)
            ordered = self._gp_bounds(bounds)
        except StoreError as exc:
            return _error(exc)
        except (KeyError, TypeError) as exc:
            return {"ok": False, "error": {"code": "invalid_request",
                                           "message": f"bounds must map each of {list(_GP_BOUND_VARS)} to [lo, hi] ({exc})"}}
        if state is None:
            return {"ok": False, "error": {"code": "invalid_request", "message": f"{fitted_model!r} has no gp_state"}}
        return self._gp_design(
            fitted_model=fitted_model, name=name, call=call, tool="discriminatory_doe_gp",
            description=description,
            arguments={"threshold": threshold, "batch_size": batch_size, "n_grid": len(grid), "seed": seed},
            run=lambda: self.doe.discriminate_gp({"model_spec": spec, "state": state, "grid": list(grid),
                                                  "threshold": threshold, "bounds": ordered,
                                                  "batch_size": batch_size, "seed": seed}),
        )

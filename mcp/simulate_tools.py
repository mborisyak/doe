"""The experiment-surrogate resolver: run a simulation, record the result as a data batch.

Kept separate from compute_tools (analysis/design) on purpose -- simulation is the data
*source*, not analysis, and a new/different simulation backend should slot in here beside
the enzyme one without touching the compute tools. Shared store helpers come from the
StoreResolver base.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from doe.store import Store, StoreError
from mcp_tools import EnzymeMcpService
from resolver_base import StoreResolver, _Abort, _error


class SimulateResolver(StoreResolver):
    def __init__(self, store: Store, enzyme_service: Optional[EnzymeMcpService] = None) -> None:
        super().__init__(store)
        self.enzyme = enzyme_service or EnzymeMcpService()

    def _design_conditions(self, design: str) -> Dict[str, Any]:
        record = self._typed(design, "design")
        return {exp: payload["conditions"] for exp, payload in record["experiments"].items()}

    def simulate(
        self,
        *,
        name: str = "-",
        design: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
        call: str = "-",
        description: str = "",
    ) -> Dict[str, Any]:
        # The sampling window, concentrations, noise, and device are config/env-owned (read by
        # the runner); the tool takes only the conditions to run (a design, or an ad-hoc map).
        try:
            if design is not None:
                conds = self._design_conditions(design)
            elif conditions is not None:
                conds = conditions
            else:
                return {"ok": False, "error": {"code": "invalid_request",
                                               "message": "simulate_enzyme_dynamics needs `design` or `conditions`."}}
            request: Dict[str, Any] = {"conditions": conds}
        except StoreError as exc:
            return _error(exc)

        try:
            # Reserve the batch + the call (simulate_enzyme_dynamics/<design>-<N>, keyed by the
            # design it ran -- or "conditions" for an ad-hoc run) before running the sim.
            target = design or "conditions"
            with self.store["data"].lock(name) as batch, \
                    self.store["logs"]["simulate_enzyme_dynamics"].lock(call, f"{target}-") as leaf:
                call_name = f"simulate_enzyme_dynamics/{leaf}"
                envelope = self.enzyme.simulate_enzyme_dynamics(request)
                if not envelope.get("ok"):
                    raise _Abort(envelope)
                payload = envelope["data"]
                experiments = {}
                for label, traj in payload["experiments"].items():
                    experiments[label] = {
                        "conditions": conds[label],
                        "measurements": {
                            "timestamps": traj["time_points"],
                            "A": traj["state_trajectories"]["A_measured"],
                        },
                    }
                outputs = {"data": batch}
                self.store["data"][batch] = {
                    "description": description,
                    "design": design,
                    "experiments": experiments,
                    "source": {"tool_result": call_name},
                }
                self.store["logs"]["simulate_enzyme_dynamics"][leaf] = self._tool_result(
                    "simulate_enzyme_dynamics",
                    {"conditions": None if design else conds},               # arguments (non-ref)
                    {"design": design},                                      # inputs (refs)
                    outputs,                                                 # outputs (produced refs)
                )
        except _Abort as abort:
            return abort.envelope
        except StoreError as exc:
            return _error(exc)
        return {"ok": True, "references": {"call": call_name, **outputs}}

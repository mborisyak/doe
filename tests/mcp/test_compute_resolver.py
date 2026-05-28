"""Resolver unit tests with stub inline services -- no jax / subprocess. They check
name resolution (refs -> the inline payload the engine consumes) and result -> committed
store records. The resolver resolves refs, locks the output, runs the (stubbed) engine,
then stamps + commits the record; the stubs just return the engine's plain result."""
from __future__ import annotations

from compute_tools import ComputeResolver
from doe.store import Store
from mcp_errors import error_response, success_response
from simulate_tools import SimulateResolver


class _StubDoe:
    def __init__(self, fit=None, propose=None):
        self.seen = {}
        self._fit = fit
        self._propose = propose

    def fit_parameters(self, request):
        self.seen["fit"] = request
        if self._fit is not None:
            return self._fit
        # The engine now returns the script's full fitted_model body; the resolver only stamps
        # linkage refs (model, fit.data, fit.tool_result, description).
        return success_response({
            "parameters": {"k": 1.5},
            "fit": {"final_loss": 0.42, "iterations": 3},
            "auxiliary": {
                "loss_trace": [0.9, 0.5, 0.42],
                "predictions": {
                    b: {"exp-1": {"timestamps": [1.0, 2.0], "A": [0.1, 0.2]}}
                    for b in request["batches"]
                },
            },
        })

    def propose_doe_experiments(self, request):
        self.seen["propose"] = request
        if self._propose is not None:
            return self._propose
        # The engine now returns the script's design body (experiments + auxiliary.expected
        # keyed by observable); the resolver stamps source + re-keys expected by fitted_model.
        return success_response({
            "experiments": {
                "exp-1": {"conditions": {"A": 1.0, "B": 2.0, "E": 0.5, "temperature": 30.0}},
                "exp-2": {"conditions": {"A": 1.5, "B": 1.0, "E": 0.6, "temperature": 40.0}},
            },
            "auxiliary": {"expected": {
                "exp-1": {"timestamps": [1.0, 2.0, 3.0], "A": [0.1, 0.2, 0.3]},
                "exp-2": {"timestamps": [1.0, 2.0, 3.0], "A": [0.4, 0.5, 0.6]},
            }},
        })


class _StubEnzyme:
    def __init__(self, sim=None):
        self.seen = {}
        self._sim = sim

    def simulate_enzyme_dynamics(self, request):
        self.seen["sim"] = request
        if self._sim is not None:
            return self._sim
        experiments = {
            label: {"time_points": [1.0, 2.0], "state_trajectories": {"A_measured": [0.9, 0.7]}}
            for label in request["conditions"]
        }
        return success_response({"experiments": experiments, "metadata": {"model_identifier": "enzyme"}})


def _model_spec():
    return {"states": ["A", "B", "E"], "parameters": {"k": [0.0, 10.0]}, "observables": {"A": "A"}}


def _seed_model(store):
    store["model"]["model-x"] = {"kind": "ode", "spec": _model_spec(), "description": "m"}


def _seed_batch(store, name="batch-1"):
    store["data"][name] = {
        "experiments": {
            "exp-1": {
                "conditions": {"A": 1.0, "B": 2.0, "E": 0.5, "temperature": 30.0},
                "measurements": {"timestamps": [1.0, 2.0], "A": [0.8, 0.6]},
            }
        },
    }


# ------------------------------------------------------------------- fit
def test_fit_resolves_refs_and_writes_fitted_model(tmp_path):
    store = Store(tmp_path / "store")
    _seed_model(store)
    _seed_batch(store, "batch-1")
    _seed_batch(store, "batch-2")
    doe = _StubDoe()
    resolver = ComputeResolver(store, doe_service=doe)

    out = resolver.fit(model="model-x", data=["batch-1", "batch-2"], description="first fit")
    assert out["ok"]
    # refs resolved to the spec + batch records the engine consumes
    req = doe.seen["fit"]
    assert req["model_spec"]["parameters"] == {"k": [0.0, 10.0]}
    assert set(req["batches"]) == {"batch-1", "batch-2"}
    assert req["batches"]["batch-1"]["experiments"]["exp-1"]["measurements"]["A"] == [0.8, 0.6]

    fm = out["references"]["fitted_model"]
    assert fm == "model-x/v1"
    rec = store.read(fm)  # committed by the resolver
    assert rec["parameters"] == {"k": 1.5}
    assert rec["model"] == "model-x" and rec["fit"]["data"] == ["batch-1", "batch-2"]
    assert rec["fit"]["final_loss"] == 0.42
    assert rec["auxiliary"]["predictions"]["batch-1"]["exp-1"]["A"] == [0.1, 0.2]
    assert out["references"]["call"] == "fit_ode/model-x-1"  # call: tool/<target>-<N>
    assert store.read("fit_ode/model-x-1")["tool"] == "fit_ode"
    assert "data" not in out  # tools return refs only, no data dump


def test_fit_missing_model_is_error(tmp_path):
    store = Store(tmp_path / "store")
    _seed_batch(store, "batch-1")
    resolver = ComputeResolver(store, doe_service=_StubDoe())
    out = resolver.fit(model="nope", data=["batch-1"])
    assert out["ok"] is False and out["error"]["code"] == "not_found"


def test_fit_compute_error_frees_the_name(tmp_path):
    store = Store(tmp_path / "store")
    _seed_model(store)
    _seed_batch(store, "batch-1")
    failing = _StubDoe(fit=error_response("numeric_instability", "diverged"))
    resolver = ComputeResolver(store, doe_service=failing)
    out = resolver.fit(model="model-x", data=["batch-1"])
    assert out["ok"] is False and out["error"]["code"] == "numeric_instability"
    assert store.names("fitted_model") == []                       # rolled back
    assert not (tmp_path / "store" / "fitted_model" / "model-x" / "v1.lock").exists()


# ------------------------------------------------------------------- propose
def test_propose_writes_design_with_expected(tmp_path):
    store = Store(tmp_path / "store")
    _seed_model(store)
    store["fitted_model"]["model-x"]["v1"] = {"model": "model-x", "parameters": {"k": 1.5}}
    resolver = ComputeResolver(store, doe_service=_StubDoe())

    out = resolver.propose(fitted_model="model-x/v1", proposal_config={"n_proposals": 2, "seed": 0})
    assert out["ok"]
    design = out["references"]["design"]
    assert design == "design-1"
    rec = store.read(design)
    assert set(rec["experiments"]) == {"exp-1", "exp-2"}
    assert rec["experiments"]["exp-1"]["conditions"]["temperature"] == 30.0
    assert rec["auxiliary"]["expected"]["model-x/v1"]["exp-2"] == {
        "timestamps": [1.0, 2.0, 3.0], "A": [0.4, 0.5, 0.6]
    }
    assert store.read("doe_ode/model-x/v1-1")["tool"] == "doe_ode"


# ------------------------------------------------------------------- simulate
def test_simulate_from_design_writes_batch(tmp_path):
    store = Store(tmp_path / "store")
    store["design"]["design-1"] = {
        "experiments": {"exp-1": {"conditions": {"A": 1.0, "B": 2.0, "E": 0.5, "temperature": 30.0}}},
    }
    enzyme = _StubEnzyme()
    resolver = SimulateResolver(store, enzyme_service=enzyme)

    out = resolver.simulate(design="design-1")
    assert out["ok"]
    assert enzyme.seen["sim"]["conditions"]["exp-1"]["A"] == 1.0
    batch = out["references"]["data"]
    assert batch == "batch-1"
    rec = store.read(batch)
    assert rec["design"] == "design-1"
    assert rec["experiments"]["exp-1"]["conditions"]["temperature"] == 30.0
    assert rec["experiments"]["exp-1"]["measurements"] == {"timestamps": [1.0, 2.0], "A": [0.9, 0.7]}


def test_simulate_requires_design_or_conditions(tmp_path):
    store = Store(tmp_path / "store")
    resolver = SimulateResolver(store, enzyme_service=_StubEnzyme())
    out = resolver.simulate()
    assert out["ok"] is False and out["error"]["code"] == "invalid_request"

"""Resolver unit tests with stub inline services -- no jax / subprocess. They check
name resolution (refs -> the core's inline payload) and result -> store records."""
from __future__ import annotations

from compute_tools import ComputeResolver
from doe.store import Store
from mcp_errors import error_response, success_response


class _StubDoe:
    def __init__(self, fit=None, propose=None):
        self.seen = {}
        self._fit = fit
        self._propose = propose

    def fit_parameters(self, request):
        self.seen["fit"] = request
        if self._fit is not None:
            return self._fit
        return success_response({
            "parameters": {"k": 1.5},
            "loss_trace": [0.9, 0.5, 0.42],
            "predictions": {label: [0.1, 0.2] for label in request["conditions"]},
        })

    def propose_doe_experiments(self, request):
        self.seen["propose"] = request
        if self._propose is not None:
            return self._propose
        return success_response({
            "proposed_conditions": [
                {"A": 1.0, "B": 2.0, "E": 0.5, "temperature": 30.0},
                {"A": 1.5, "B": 1.0, "E": 0.6, "temperature": 40.0},
            ],
            "proposal_timestamps": [1.0, 2.0, 3.0],
            "expected": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
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
    store["batch"][name] = {
        "space": ["A", "B", "E", "temperature"],
        "observable": "A_measured",
        "experiments": {"experiment-1": {"x": {"A": 1.0, "B": 2.0, "E": 0.5, "temperature": 30.0}}},
        "auxiliary": {"experiment-1": {"t": [1.0, 2.0], "y": [0.8, 0.6]}},
    }


# ------------------------------------------------------------------- fit
def test_fit_resolves_refs_and_writes_fitted_model(tmp_path):
    store = Store(tmp_path / "store")
    _seed_model(store)
    _seed_batch(store, "batch-1")
    _seed_batch(store, "batch-2")
    doe = _StubDoe()
    resolver = ComputeResolver(store, doe_service=doe, enzyme_service=_StubEnzyme())

    out = resolver.fit(model="model-x", data=["batch-1", "batch-2"], description="first fit")
    assert out["ok"]
    req = doe.seen["fit"]
    assert req["model_spec"]["parameters"] == {"k": [0.0, 10.0]}
    assert set(req["conditions"]) == {"batch-1/experiment-1", "batch-2/experiment-1"}
    assert req["measurements"]["batch-1/experiment-1"] == {"timestamps": [1.0, 2.0], "measurements": [0.8, 0.6]}

    fm = out["references"]["fitted_model"]
    assert fm == "model-x/v1"
    rec = store.read(fm)
    assert rec["parameters"] == {"k": 1.5}
    assert rec["model"] == "model-x" and rec["fit"]["data"] == ["batch-1", "batch-2"]
    assert rec["fit"]["final_loss"] == 0.42
    assert rec["auxiliary"]["loss_trace"] == [0.9, 0.5, 0.42]
    assert out["references"]["call"] == "fit/model-x/v1"
    assert store.read("fit/model-x/v1")["tool"] == "fit_parameters"
    # the returned data nulls auxiliary (wire view)
    assert out["data"]["auxiliary"] == {"loss_trace": None, "predictions": None}


def test_fit_missing_model_is_error(tmp_path):
    store = Store(tmp_path / "store")
    _seed_batch(store, "batch-1")
    resolver = ComputeResolver(store, doe_service=_StubDoe(), enzyme_service=_StubEnzyme())
    out = resolver.fit(model="nope", data=["batch-1"])
    assert out["ok"] is False and out["error"]["code"] == "not_found"


def test_fit_compute_error_frees_the_name(tmp_path):
    store = Store(tmp_path / "store")
    _seed_model(store)
    _seed_batch(store, "batch-1")
    failing = _StubDoe(fit=error_response("numeric_instability", "diverged"))
    resolver = ComputeResolver(store, doe_service=failing, enzyme_service=_StubEnzyme())
    out = resolver.fit(model="model-x", data=["batch-1"])
    assert out["ok"] is False and out["error"]["code"] == "numeric_instability"
    assert store.names("fitted_model") == []                       # rolled back
    assert not (tmp_path / "store" / "fitted_models" / "model-x" / "v1.lock").exists()


# ------------------------------------------------------------------- propose
def test_propose_writes_design_with_expected(tmp_path):
    store = Store(tmp_path / "store")
    _seed_model(store)
    store["fitted_model"]["model-x"]["v1"] = {"model": "model-x", "parameters": {"k": 1.5}}
    resolver = ComputeResolver(store, doe_service=_StubDoe(), enzyme_service=_StubEnzyme())

    out = resolver.propose(fitted_model="model-x/v1", proposal_config={"n_proposals": 2, "seed": 0})
    assert out["ok"]
    design = out["references"]["design"]
    assert design == "design-1"
    rec = store.read(design)
    assert rec["observable"] == "A"
    assert set(rec["experiments"]) == {"experiment-1", "experiment-2"}
    assert rec["experiments"]["experiment-1"]["x"]["temperature"] == 30.0
    assert rec["auxiliary"]["expected"]["model-x/v1"]["experiment-2"] == [0.4, 0.5, 0.6]
    assert rec["auxiliary"]["experiment-1"]["t"] == [1.0, 2.0, 3.0]
    assert store.read("propose/design-1")["tool"] == "propose_doe_experiments"


# ------------------------------------------------------------------- simulate
def test_simulate_from_design_writes_batch(tmp_path):
    store = Store(tmp_path / "store")
    store["design"]["design-1"] = {
        "space": ["A", "B", "E", "temperature"],
        "experiments": {"experiment-1": {"x": {"A": 1.0, "B": 2.0, "E": 0.5, "temperature": 30.0}}},
    }
    enzyme = _StubEnzyme()
    resolver = ComputeResolver(store, doe_service=_StubDoe(), enzyme_service=enzyme)

    out = resolver.simulate(design="design-1", time={"t_start": 0.0, "t_end": 5.0, "measurements": 2})
    assert out["ok"]
    assert enzyme.seen["sim"]["conditions"]["experiment-1"]["A"] == 1.0
    batch = out["references"]["batch"]
    assert batch == "batch-1"
    rec = store.read(batch)
    assert rec["design"] == "design-1"
    assert rec["experiments"]["experiment-1"]["x"]["temperature"] == 30.0
    assert rec["auxiliary"]["experiment-1"]["y"] == [0.9, 0.7]


def test_simulate_requires_design_or_conditions(tmp_path):
    store = Store(tmp_path / "store")
    resolver = ComputeResolver(store, doe_service=_StubDoe(), enzyme_service=_StubEnzyme())
    out = resolver.simulate()
    assert out["ok"] is False and out["error"]["code"] == "invalid_request"

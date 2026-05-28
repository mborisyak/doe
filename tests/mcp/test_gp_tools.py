"""GP modelling tools (create_gp / fit_gp / predict_gp / hyper_fit_gp), in-process."""
from __future__ import annotations

import numpy as np

from compute_tools import ComputeResolver
from doe.store import Store


def _seed_batch(store, name, conditions):
    """A batch whose experiments carry an A(t) decay curve, so the GP has (conditions,t)->A data."""
    timestamps = [2.0, 4.0, 6.0, 8.0]
    experiments = {}
    for i, (a, b, e, temp) in enumerate(conditions):
        ys = [float(a * np.exp(-0.1 * (1.0 + 0.01 * temp) * t)) for t in timestamps]
        experiments[f"exp-{i + 1}"] = {
            "conditions": {"A": a, "B": b, "E": e, "temperature": temp},
            "measurements": {"timestamps": timestamps, "A": ys},
        }
    store["data"][name] = {"experiments": experiments}


_CONDS = [(2.0, 2.0, 0.002, 37.0), (1.0, 3.0, 0.001, 30.0), (3.0, 1.0, 0.003, 40.0), (1.5, 2.5, 0.0015, 33.0)]


def test_create_fit_predict_gp(tmp_path):
    store = Store(tmp_path / "store")
    resolver = ComputeResolver(store)
    _seed_batch(store, "batch-1", _CONDS)

    # create_gp -> a kind:"gp" model
    out = resolver.create_gp(name="gp-1", length_scale=10.0, variance=1.0, noise=0.05, observables=["A"])
    assert out["ok"] and out["references"]["model"] == "gp-1"
    model = store.read("gp-1")
    assert model["kind"] == "gp" and model["spec"]["observables"] == ["A"]

    # fit_gp -> a fitted_model carrying the serialised GP state + predictions
    out = resolver.fit_gp(model="gp-1", data=["batch-1"])
    assert out["ok"]
    fm = out["references"]["fitted_model"]
    assert fm == "gp-1/v1"
    rec = store.read(fm)
    assert set(rec["auxiliary"]["gp_state"]) == {"X_flat", "L", "alpha"}
    assert rec["auxiliary"]["predictions"]["A"]["mean"]  # keyed by observable
    assert rec["fit"]["cv_rmse"] is not None
    assert store.read("fit_gp/gp-1-1")["tool"] == "fit_gp"  # call: tool/<target>-<N>
    assert "data" not in out  # tools return refs only, no data dump

    # predict_gp at explicit [A, B, E, temperature, t] points -> {mean, std}
    out = resolver.predict_gp(fitted_model=fm, points=[[2.0, 2.0, 0.002, 37.0, 5.0],
                                                       [1.0, 3.0, 0.001, 30.0, 3.0]])
    assert out["ok"]
    pred = out["data"]["A"]
    assert len(pred["mean"]) == 2 and len(pred["std"]) == 2
    assert all(s >= 0 for s in pred["std"])


def test_fit_gp_missing_model_is_error(tmp_path):
    store = Store(tmp_path / "store")
    resolver = ComputeResolver(store)
    _seed_batch(store, "batch-1", _CONDS)
    out = resolver.fit_gp(model="nope", data=["batch-1"])
    assert out["ok"] is False and out["error"]["code"] == "not_found"


def test_hyper_fit_gp_optimizes_length_scales(tmp_path):
    store = Store(tmp_path / "store")
    resolver = ComputeResolver(store)
    _seed_batch(store, "batch-1", _CONDS)

    out = resolver.hyper_fit_gp(data=["batch-1"], name="gp-h", observables=["A"], noise=0.05)
    assert out["ok"] and out["references"]["model"] == "gp-h"
    spec = store.read("gp-h")["spec"]
    assert spec["kernel"] == "rbf"
    assert len(spec["length_scales"]) == 5  # 4 conditions + time
    assert all(np.isfinite(spec["length_scales"])) and spec["variance"] > 0

    # the optimised model is then usable by fit_gp
    out = resolver.fit_gp(model="gp-h", data=["batch-1"])
    assert out["ok"]


_BOUNDS = {"A": [1.0, 3.0], "B": [1.0, 3.0], "E": [0.001, 0.003], "temperature": [25.0, 45.0], "t": [2.0, 8.0]}


def _fitted_gp(tmp_path):
    store = Store(tmp_path / "store")
    resolver = ComputeResolver(store)
    _seed_batch(store, "batch-1", _CONDS)
    resolver.create_gp(name="gp-1", length_scale=10.0, variance=1.0, noise=0.05, observables=["A"])
    fm = resolver.fit_gp(model="gp-1", data=["batch-1"])["references"]["fitted_model"]
    return store, resolver, fm


def test_doe_gp_writes_design(tmp_path):
    store, resolver, fm = _fitted_gp(tmp_path)
    out = resolver.doe_gp(fitted_model=fm, bounds=_BOUNDS, batch_size=3, seed=0)
    assert out["ok"]
    design = out["references"]["design"]
    assert design == "design-1"
    rec = store.read(design)
    assert len(rec["experiments"]) == 3
    exp1 = rec["experiments"]["exp-1"]["conditions"]
    assert set(exp1) == {"A", "B", "E", "temperature"}  # t lives in the expected, not conditions
    assert _BOUNDS["A"][0] <= exp1["A"] <= _BOUNDS["A"][1]
    assert rec["auxiliary"]["eig"] >= 0.0
    expected = rec["auxiliary"]["expected"][fm]["exp-1"]
    assert "A" in expected and "sigma_A" in expected and len(expected["timestamps"]) == 1
    assert store.read("doe_gp/gp-1/v1-1")["tool"] == "doe_gp"


def test_discriminatory_doe_gp_writes_design(tmp_path):
    store, resolver, fm = _fitted_gp(tmp_path)
    grid = [[2.0, 2.0, 0.002, 37.0, t] for t in (2.0, 4.0, 6.0, 8.0)]
    out = resolver.discriminatory_doe_gp(
        fitted_model=fm, grid=grid, threshold=1.0, bounds=_BOUNDS, batch_size=2, seed=0,
    )
    assert out["ok"]
    design = out["references"]["design"]
    rec = store.read(design)
    assert len(rec["experiments"]) == 2
    assert "eig" in rec["auxiliary"]
    assert store.read("discriminatory_doe_gp/gp-1/v1-1")["tool"] == "discriminatory_doe_gp"


def test_doe_gp_bad_bounds_is_error(tmp_path):
    store, resolver, fm = _fitted_gp(tmp_path)
    out = resolver.doe_gp(fitted_model=fm, bounds={"A": [1.0, 3.0]}, batch_size=2, seed=0)  # missing dims
    assert out["ok"] is False and out["error"]["code"] == "invalid_request"

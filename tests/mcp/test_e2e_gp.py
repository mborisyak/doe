"""End-to-end GP workflow through the actual MCP tools.

Mirrors tests/mcp/test_e2e.py (loads the real server.py against a fixed store and drives
the registered @tool functions), but for the GP track: an agent writes synthetic data of
a known smooth field A(conditions, t), hyper-fits + fits a GP, checks predictions behave
(uncertainty grows away from data), then runs both GP DoE tools. The raw store API is used
only for the agent-side data write and the final inspection. GP tools are in-process so
this is fast (no enzyme simulation).
"""
from __future__ import annotations

import runpy
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Fixed, git-ignored store so the produced records can be inspected after the run.
STORE_ROOT = ROOT / "tests" / "e2e_gp_store"

BOUNDS = {"A": [1.0, 3.0], "B": [1.0, 3.0], "E": [0.001, 0.003], "temperature": [25.0, 45.0], "t": [2.0, 10.0]}


def _field(a: float, temperature: float, t: float) -> float:
    # smooth in A, temperature, t; independent of B, E (the GP should down-weight them).
    return float(a * np.exp(-0.08 * t) + 0.01 * temperature)


def test_end_to_end_gp_through_mcp_tools(monkeypatch):
    if STORE_ROOT.exists():
        shutil.rmtree(STORE_ROOT)
    monkeypatch.setenv("DOE_STORE_ROOT", str(STORE_ROOT))
    ns = runpy.run_path(str(ROOT / "mcp" / "server.py"), run_name="not_main")
    store = ns["store"]
    get_config = ns["get_config"]
    hyper_fit_gp, fit_gp, predict_gp = ns["hyper_fit_gp"], ns["fit_gp"], ns["predict_gp"]
    doe_gp, discriminatory_doe_gp = ns["doe_gp"], ns["discriminatory_doe_gp"]

    assert get_config()["ok"]

    # 1. agent writes synthetic data: A(conditions, t) over a smooth field, via the store API
    rng = np.random.default_rng(0)
    timestamps = [2.0, 4.0, 6.0, 8.0, 10.0]
    experiments = {}
    for i in range(5):
        a = float(rng.uniform(1.0, 3.0))
        temperature = float(rng.uniform(25.0, 45.0))
        experiments[f"exp-{i + 1}"] = {
            "conditions": {"A": a, "B": 2.0, "E": 0.002, "temperature": temperature},
            "measurements": {"timestamps": timestamps, "A": [_field(a, temperature, t) for t in timestamps]},
        }
    store["data"]["batch-1"] = {"experiments": experiments, "description": "synthetic GP field"}

    # 2. hyper-fit a GP model to the data (per-dimension length scales), then 3. fit it
    out = hyper_fit_gp(data=["batch-1"], name="gp-1", observables=["A"], noise=0.02)
    assert out["ok"], out
    assert len(store.read("gp-1")["spec"]["length_scales"]) == 5

    out = fit_gp(model="gp-1", data=["batch-1"])
    assert out["ok"], out
    fm = out["references"]["fitted_model"]
    assert np.isfinite(store.read(fm)["fit"]["cv_rmse"])

    # 4. predict: a point inside the data is far more certain than one far outside it
    near = [2.0, 2.0, 0.002, 35.0, 4.0]
    far = [50.0, 2.0, 0.002, 35.0, 100.0]
    out = predict_gp(fitted_model=fm, points=[near, far])
    assert out["ok"], out
    std = out["data"]["A"]["std"]
    assert all(s >= 0 for s in std)
    assert std[1] > std[0]

    # 5. batch-BALD DoE -> a design with the GP's expected output per proposed point
    out = doe_gp(fitted_model=fm, bounds=BOUNDS, batch_size=3, name="doe-design")
    assert out["ok"], out
    d1 = out["references"]["design"]
    rec1 = store.read(d1)
    assert len(rec1["experiments"]) == 3
    assert rec1["auxiliary"]["eig"] >= 0.0
    assert "A" in rec1["auxiliary"]["expected"][fm]["exp-1"]
    assert store.read("doe_gp/gp-1/v1-1")["tool"] == "doe_gp"  # call: tool/<fitted_model>-<N>

    # 6. discriminative DoE over a grid in (conditions, t) -> a design
    grid = [[a, 2.0, 0.002, 35.0, t] for a in (1.0, 2.0, 3.0) for t in (2.0, 6.0, 10.0)]
    out = discriminatory_doe_gp(
        fitted_model=fm, grid=grid, threshold=1.0, bounds=BOUNDS, batch_size=2, name="disc-design",
    )
    assert out["ok"], out
    d2 = out["references"]["design"]
    assert len(store.read(d2)["experiments"]) == 2
    assert store.read("discriminatory_doe_gp/gp-1/v1-1")["tool"] == "discriminatory_doe_gp"

    # 7. inspect the accumulated store via the Python API
    assert "gp-1" in store.names("model")
    assert fm in store.names("fitted_model")
    assert {d1, d2} <= set(store.names("design"))

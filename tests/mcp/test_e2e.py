"""Minimal end-to-end workflow through the actual MCP tools.

Loads the real MCP server (server.py) against a temp store and drives the registered
tools an agent would call -- get_config, simulate_enzyme_dynamics, store_create,
fit_ode, doe_ode -- with the raw Python store API used only where
an agent would reach past the tools: writing the initial design and the final comparison.
Runs the real scripts + enzyme simulator; slow (several JAX fits + simulations).
"""
from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models"
# Fixed, git-ignored store so the produced records can be inspected after the run.
STORE_ROOT = ROOT / "tests" / "e2e_store"


def _spec(spec_file: str) -> dict:
    with (MODELS / spec_file).open() as stream:
        return json.load(stream)


@pytest.mark.skipif(
    not (MODELS / "super-model-1.json").is_file() or not (MODELS / "super-model-3.json").is_file(),
    reason="model specs not present",
)
def test_end_to_end_through_mcp_tools(monkeypatch):
    # Wipe the fixed store first -- it's append-only, so a rerun would collide on names --
    # then load the real server module bound to it; its @tool functions are the MCP interface.
    if STORE_ROOT.exists():
        shutil.rmtree(STORE_ROOT)
    monkeypatch.setenv("DOE_STORE_ROOT", str(STORE_ROOT))
    ns = runpy.run_path(str(ROOT / "mcp" / "server.py"), run_name="not_main")
    store = ns["store"]  # the Python store API, for the agent-side bits
    get_config = ns["get_config"]
    simulate = ns["simulate_enzyme_dynamics"]
    store_create = ns["store_create"]
    fit_ode = ns["fit_ode"]
    propose = ns["doe_ode"]

    # 0. agent reads the experiment config (bounds, durations) via the MCP tool
    cfg = get_config()
    assert cfg["ok"] and cfg["data"]["name"] == "enzyme"

    # 1. agent writes a random design straight through the store API
    rng = np.random.default_rng(0)
    store["design"]["design-1"] = {
        "experiments": {
            f"exp-{i + 1}": {"conditions": {
                "A": float(rng.uniform(1.0, 3.0)),
                "B": float(rng.uniform(1.0, 3.0)),
                "E": float(rng.uniform(1.0e-3, 3.0e-3)),
                "temperature": float(rng.uniform(25.0, 45.0)),
            }}
            for i in range(2)
        },
        "description": "random baseline",
    }

    # 2. simulate it (MCP tool) -> first data batch
    out = simulate(design="design-1")
    assert out["ok"], out
    batch1 = out["references"]["data"]

    # 3. register a candidate model (MCP tool) and fit it (MCP tool)
    out = store_create(type="model", name="super-model-1", record={"kind": "ode", "spec": _spec("super-model-1.json")})
    assert out["ok"], out
    out = fit_ode(model="super-model-1", data=[batch1], optimizer={"iterations": 60})
    assert out["ok"], out
    fit1 = out["references"]["fitted_model"]
    assert fit1 == "super-model-1/v1"

    # 4. propose a DoE (MCP tool), simulate it -> second batch
    out = propose(fitted_model=fit1, proposal_config={"n_proposals": 2, "iterations": 20, "seed": 0})
    assert out["ok"], out
    design2 = out["references"]["design"]
    out = simulate(design=design2)
    assert out["ok"], out
    batch2 = out["references"]["data"]

    # 5. refit super-model-1 on both batches
    out = fit_ode(model="super-model-1", data=[batch1, batch2], optimizer={"iterations": 60})
    assert out["ok"], out
    fit1_v2 = out["references"]["fitted_model"]
    assert fit1_v2 == "super-model-1/v2"

    # 6. register + fit a second candidate model
    out = store_create(type="model", name="super-model-3", record={"kind": "ode", "spec": _spec("super-model-3.json")})
    assert out["ok"], out
    out = fit_ode(model="super-model-3", data=[batch1, batch2], optimizer={"iterations": 60})
    assert out["ok"], out
    fit3 = out["references"]["fitted_model"]

    # 7. compare the two fitted models by hand, via the store API
    loss_one = store.read(fit1_v2)["fit"]["final_loss"]
    loss_three = store.read(fit3)["fit"]["final_loss"]
    assert np.isfinite(loss_one) and np.isfinite(loss_three)
    assert (fit1_v2 if loss_one <= loss_three else fit3) in {fit1_v2, fit3}

    # the store accumulated the whole workflow, append-only
    assert set(store.names("data")) == {batch1, batch2}
    assert {fit1, fit1_v2, fit3} <= set(store.names("fitted_model"))
    assert set(store.names("design")) == {"design-1", design2}

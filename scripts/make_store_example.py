"""Regenerate ``store-example/`` -- a small, hand-curated illustration of the store layout.

Unlike ``tests/e2e_store`` (produced by running the real tools), this is a deliberately
tiny teaching store with clean round numbers, written straight through the ``doe.store``
Python API so the on-disk layout and stamps are exactly what the server produces:

  * singular collection dirs (model/ fitted_model/ design/ data/ logs/),
  * ``logs`` records split into {tool, status, arguments, inputs, outputs},
  * call names ``<tool>/<target>-<N>`` (keyed by the input the tool acted on),
  * ODE fit/propose outputs carry no predictive sigma (only the GP track does).

Run:  python scripts/make_store_example.py [ROOT]   (default ROOT = store-example)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from doe.store import Store

TS = [2.0, 4.0, 6.0]  # shared illustrative measurement timestamps


def build(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    store = Store(root)

    # --- models (structure only; parameter *ranges*, not values) -----------------
    store["model"]["model-x"] = {
        "kind": "ode",
        "description": "sequential ordered ternary-complex mechanism",
        "spec": {
            "states": ["A", "B", "C", "D", "E", "EA", "EAB", "ECD"],
            "parameters": {"k1_on": [1.0, 1000.0], "k1_off": [0.01, 100.0], "kcat": [0.1, 100.0]},
            "initial_state": {"A": "A0", "B": "B0", "E": "E0"},
            "algebraics": {"r1_on": "k1_on * A * E", "r3_cat": "kcat * EAB"},
            "rhs": {"A": "-r1_on", "E": "-r1_on"},
            "observables": {"A": "A"},
        },
    }
    store["model"]["model-y"] = {
        "kind": "gp",
        "description": "RBF GP surrogate over (A, B, E, temperature, t)",
        "spec": {"kernel": "rbf", "length_scales": [2.0, 2.0, 0.5, 8.0, 4.0],
                 "variance": 1.0, "noise": 0.02, "observables": ["A"]},
    }

    # --- batch-1: a random baseline, simulated from ad-hoc conditions (no design) ---
    store["data"]["batch-1"] = {
        "description": "random baseline simulation",
        "design": None,
        "experiments": {"exp-1": {
            "conditions": {"A": 5.5, "B": 7.2, "E": 0.5, "temperature": 37.0},
            "measurements": {"timestamps": TS, "A": [0.91, 0.78, 0.66]},
        }},
        "source": {"tool_result": "simulate_enzyme_dynamics/conditions-1"},
    }

    # --- model-x/v1: MLE fit on batch-1 (ODE preds: timestamps + observable, no sigma) ---
    store["fitted_model"]["model-x"]["v1"] = {
        "description": "fit on the baseline batch",
        "model": "model-x",
        "parameters": {"k1_on": 13.7, "k1_off": 0.10, "kcat": 8.9},
        "fit": {"data": ["batch-1"], "tool_result": "fit_ode/model-x-1",
                "final_loss": 0.27, "iterations": 60},
        "auxiliary": {
            "loss_trace": [0.29, 0.27, 0.27],
            "predictions": {"batch-1": {"exp-1": {"timestamps": TS, "A": [0.95, 0.88, 0.81]}}},
        },
    }

    # --- design-1: Fisher D-optimal proposal under model-x/v1 (expected, no sigma) ---
    store["design"]["design-1"] = {
        "description": "D-optimal proposal under model-x/v1",
        "experiments": {
            "exp-1": {"conditions": {"A": 1.5, "B": 2.0, "E": 1.2, "temperature": 37.0}},
            "exp-2": {"conditions": {"A": 0.5, "B": 3.0, "E": 0.8, "temperature": 30.0}},
        },
        "auxiliary": {"expected": {"model-x/v1": {
            "exp-1": {"timestamps": TS, "A": [0.90, 0.71, 0.55]},
            "exp-2": {"timestamps": TS, "A": [0.81, 0.62, 0.47]},
        }}},
        "source": {"tool_result": "doe_ode/model-x/v1-1", "model": "model-x/v1"},
    }

    # --- batch-2: simulate the proposed design (executed conditions may drift from design) ---
    store["data"]["batch-2"] = {
        "description": "data collected for the DoE proposal",
        "design": "design-1",
        "experiments": {
            "exp-1": {"conditions": {"A": 1.5, "B": 2.0, "E": 1.2, "temperature": 37.0},
                      "measurements": {"timestamps": TS, "A": [0.89, 0.69, 0.52]}},
            "exp-2": {"conditions": {"A": 0.5, "B": 3.0, "E": 0.8, "temperature": 30.0},
                      "measurements": {"timestamps": TS, "A": [0.83, 0.64, 0.50]}},
        },
        "source": {"tool_result": "simulate_enzyme_dynamics/design-1-1"},
    }

    # --- model-x/v2: refit on both batches ---------------------------------------
    store["fitted_model"]["model-x"]["v2"] = {
        "description": "refit after adding the DoE batch",
        "model": "model-x",
        "parameters": {"k1_on": 12.3, "k1_off": 0.098, "kcat": 9.4},
        "fit": {"data": ["batch-1", "batch-2"], "tool_result": "fit_ode/model-x-2",
                "final_loss": 0.13, "iterations": 60},
        "auxiliary": {
            "loss_trace": [0.21, 0.15, 0.13],
            "predictions": {
                "batch-1": {"exp-1": {"timestamps": TS, "A": [0.92, 0.80, 0.69]}},
                "batch-2": {"exp-1": {"timestamps": TS, "A": [0.88, 0.70, 0.54]},
                            "exp-2": {"timestamps": TS, "A": [0.82, 0.63, 0.49]}},
            },
        },
    }

    # --- logs: call records {tool, status, arguments(non-ref), inputs(refs), outputs(refs)} ---
    store["logs"]["simulate_enzyme_dynamics"]["conditions-1"] = {
        "tool": "simulate_enzyme_dynamics", "status": "ok",
        "arguments": {"conditions": {"exp-1": {"A": 5.5, "B": 7.2, "E": 0.5, "temperature": 37.0}}},
        "inputs": {}, "outputs": {"data": "batch-1"},
    }
    store["logs"]["fit_ode"]["model-x-1"] = {
        "tool": "fit_ode", "status": "ok",
        "arguments": {"optimizer": {"iterations": 60}},
        "inputs": {"model": "model-x", "data": ["batch-1"]},
        "outputs": {"fitted_model": "model-x/v1"},
    }
    store["logs"]["doe_ode"]["model-x/v1-1"] = {
        "tool": "doe_ode", "status": "ok",
        "arguments": {"proposal_config": {"n_proposals": 2, "criterion": "D", "seed": 0}},
        "inputs": {"fitted_model": "model-x/v1"},
        "outputs": {"design": "design-1"},
    }
    store["logs"]["simulate_enzyme_dynamics"]["design-1-1"] = {
        "tool": "simulate_enzyme_dynamics", "status": "ok",
        "arguments": {}, "inputs": {"design": "design-1"}, "outputs": {"data": "batch-2"},
    }
    store["logs"]["fit_ode"]["model-x-2"] = {
        "tool": "fit_ode", "status": "ok",
        "arguments": {"optimizer": {"iterations": 60}},
        "inputs": {"model": "model-x", "data": ["batch-1", "batch-2"]},
        "outputs": {"fitted_model": "model-x/v2"},
    }
    # a free-form log: the agent's own note, any `tool` name, any body.
    store["logs"]["result-1"] = {
        "tool": "model_comparison",
        "description": "AIC says v2 beats v1; stop refining structure",
        "winner": "model-x/v2",
        "aic": {"model-x/v1": 120.4, "model-x/v2": 98.1},
    }


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("store-example")
    build(root)
    print(f"wrote example store to {root}/")


if __name__ == "__main__":
    main()

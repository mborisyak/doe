"""GP surrogate glue for the store-free GP scripts.

A GP **model spec** (``kind: "gp"``) gives the kernel + hyper-parameters and the
observable list::

    {"kernel": "rbf", "length_scales": [...], "variance": 1.0, "noise": 0.01,
     "observables": ["A"]}

The surrogate maps ``(conditions, t) -> scalar`` for one observable: input ``X`` is the
condition variables (``A, B, E, temperature``) plus the timestamp ``t`` as a final
dimension, target ``y`` is the observable's measured value at that ``t``. Multiple
observables are not supported yet -- :func:`observable_of` asserts exactly one, and the
scripts wrap every output under that observable's name so the on-disk shape is already
multi-output-ready (``{<obs>: ...}``).

``build_kernel`` reconstructs the kernel callable; ``state_to_json`` / ``state_from_json``
(de)serialise a fitted :class:`doe.gp.gp.State` for storage in a ``fitted_model``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import jax.numpy as jnp

from . import kernels
from .gp import GP, State

# GP input layout: the condition variables, then time as the last coordinate. Hardcoded
# (rather than doe.common.Conditions._fields) so this module -- and the subprocess scripts
# that import it -- don't drag in the sympy/ODE stack just for four names.
INPUT_VARS: Tuple[str, ...] = ("A", "B", "E", "temperature")
INPUT_DIM = len(INPUT_VARS) + 1


def observable_of(spec: Dict[str, Any]) -> str:
    """The single observable this GP surrogates. Multi-output GP is not implemented yet,
    so more than one observable is rejected (the scripts still key outputs by this name)."""
    observables = spec.get("observables")
    if isinstance(observables, dict):  # ODE-style {name: expr}; accept its keys too
        observables = list(observables)
    if not isinstance(observables, list) or not observables:
        raise ValueError("GP spec must list at least one observable")
    assert len(observables) == 1, (
        f"multi-output GP not supported yet: spec lists {len(observables)} observables {observables}"
    )
    return observables[0]


def build_kernel(spec: Dict[str, Any]):
    """Build the kernel callable from a GP spec. `length_scales` (a vector, per-dimension)
    or `length_scale` (a scalar); only `rbf` supports a vector scale."""
    name = spec.get("kernel", "rbf")
    variance = float(spec.get("variance", 1.0))
    if "length_scales" in spec:
        length_scale: Any = jnp.asarray(spec["length_scales"], jnp.float32)
    else:
        length_scale = float(spec.get("length_scale", 1.0))
    if name == "rbf":
        return kernels.rbf(length_scale=length_scale, variance=variance)
    if name in ("matern52", "matern32"):
        if jnp.ndim(length_scale) != 0:
            raise ValueError(f"{name} takes a scalar length_scale; use 'rbf' for per-dimension scales")
        return getattr(kernels, name)(length_scale=float(length_scale), variance=variance)
    raise ValueError(f"unknown kernel {name!r}")


def make_gp(spec: Dict[str, Any]) -> GP:
    return GP(build_kernel(spec), noise=float(spec.get("noise", 1.0e-6)))


def dataset_xy(dataset, observable: str, input_vars: Sequence[str] = INPUT_VARS):
    """Flatten an assembled :class:`doe.dataset.Dataset` to GP training data: one
    ``(X, y)`` row per (experiment, timestamp), ``X = [*conditions, t]``, ``y`` the
    observable's value at ``t``. Skips experiments that don't measure ``observable``."""
    rows: List[List[float]] = []
    targets: List[float] = []
    for entry in dataset.entries.values():
        measurements = entry.get("measurements") or {}
        if observable not in measurements:
            continue
        conditions = entry["conditions"]
        base = [float(conditions[v]) for v in input_vars]
        for t, value in zip(measurements["timestamps"], measurements[observable]):
            rows.append(base + [float(t)])
            targets.append(float(value))
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def state_to_json(state: State) -> Dict[str, Any]:
    return {
        "X_flat": np.asarray(state.X_flat).tolist(),
        "L": np.asarray(state.L).tolist(),
        "alpha": np.asarray(state.alpha).tolist(),
    }


def state_from_json(payload: Dict[str, Any]) -> State:
    return State(
        X_flat=jnp.asarray(payload["X_flat"], jnp.float32),
        L=jnp.asarray(payload["L"], jnp.float32),
        alpha=jnp.asarray(payload["alpha"], jnp.float32),
    )


def _cross_val_rmse(gp: GP, X, y, folds: int, seed: int = 0):
    n = len(y)
    if n < 2:
        return None
    k = max(2, min(folds, n))
    order = np.random.default_rng(seed).permutation(n)
    predicted = np.empty(n)
    for fold in range(k):
        val = order[fold::k]
        train = np.setdiff1d(order, val, assume_unique=False)
        if len(train) == 0 or len(val) == 0:
            return None
        state = gp.fit(jnp.asarray(X[train], jnp.float32), jnp.asarray(y[train], jnp.float32))
        mean, _ = gp.predict(state, jnp.asarray(X[val], jnp.float32))
        predicted[val] = np.asarray(mean)
    return float(np.sqrt(np.mean((predicted - y) ** 2)))


def fit_surrogate(spec: Dict[str, Any], X, y, folds: int = 5) -> Dict[str, Any]:
    """Cross-validate (k-fold RMSE) and refit a GP with the spec's fixed hyper-parameters
    on ``(X, y)``; return ``{observable, n_train, cv_rmse, state, predictions}`` with the
    refit State serialised and the latent mean/std at the training inputs, keyed by the
    single observable name (multi-output-ready shape)."""
    observable = observable_of(spec)
    gp = make_gp(spec)
    cv_rmse = _cross_val_rmse(gp, X, y, folds)
    state = gp.fit(jnp.asarray(X, jnp.float32), jnp.asarray(y, jnp.float32))
    mean, var = gp.predict(state, jnp.asarray(X, jnp.float32))
    std = np.sqrt(np.clip(np.asarray(var), 0.0, None))
    return {
        "observable": observable,
        "n_train": int(len(y)),
        "cv_rmse": cv_rmse,
        "state": state_to_json(state),
        "predictions": {observable: {"mean": [float(v) for v in np.asarray(mean)],
                                     "std": [float(v) for v in std]}},
    }


def predict_surrogate(spec: Dict[str, Any], state_payload: Dict[str, Any], X_points) -> Dict[str, Any]:
    """Latent mean + std of the fitted GP at explicit input points ``X_points`` (N, D),
    keyed by the single observable name."""
    observable = observable_of(spec)
    gp = make_gp(spec)
    state = state_from_json(state_payload)
    mean, var = gp.predict(state, jnp.asarray(np.asarray(X_points, dtype=np.float64), jnp.float32))
    std = np.sqrt(np.clip(np.asarray(var), 0.0, None))
    return {observable: {"mean": [float(v) for v in np.asarray(mean)],
                         "std": [float(v) for v in std]}}


def design_body(spec: Dict[str, Any], state_payload: Dict[str, Any], B, eig: float) -> Dict[str, Any]:
    """Assemble the design record body for a GP DoE batch ``B`` (rows of ``[*conditions, t]``):
    the proposed ``experiments`` (conditions only -- ``t`` lives in the expected) plus
    ``auxiliary`` holding the GP's expected output per experiment (mean + sigma at the row's
    timestamp, keyed by the observable) and the batch's expected information gain ``eig``.
    The MCP layer dumps this and re-keys ``auxiliary.expected`` by the fitted_model it ran on."""
    observable = observable_of(spec)
    preds = predict_surrogate(spec, state_payload, B)[observable]
    means, stds = preds["mean"], preds["std"]
    n_cond = len(INPUT_VARS)
    experiments, expected = {}, {}
    for i, row in enumerate(B):
        label = f"exp-{i + 1}"
        experiments[label] = {"conditions": {v: float(row[j]) for j, v in enumerate(INPUT_VARS)}}
        expected[label] = {
            "timestamps": [float(row[n_cond])],
            observable: [float(means[i])],
            f"sigma_{observable}": [float(stds[i])],
        }
    return {"experiments": experiments, "auxiliary": {"expected": expected, "eig": float(eig)}}


def optimise_hyperparameters(spec: Dict[str, Any], X, y, seed: int = 0) -> Dict[str, Any]:
    """Return a copy of ``spec`` with per-dimension RBF ``length_scales`` and ``variance``
    re-fit to ``(X, y)`` by maximising the GP log-marginal-likelihood (multi-start
    L-BFGS-B over log-hyper-parameters); ``noise`` is kept as given."""
    import scipy.optimize as spopt
    import jax

    X_j = jnp.asarray(X, jnp.float32)
    y_j = jnp.asarray(y, jnp.float32)
    dim = X_j.shape[1]
    noise = float(spec.get("noise", 1.0e-6))

    def neg_lml(theta):
        ls, var = jnp.exp(theta[:dim]), jnp.exp(theta[dim])
        gp = GP(kernels.rbf(length_scale=ls, variance=var), noise=noise)
        return -gp.log_marginal_likelihood(X_j, y_j)

    value_and_grad = jax.jit(jax.value_and_grad(neg_lml))

    def fun(theta):
        v, g = value_and_grad(jnp.asarray(theta))
        return float(v), np.asarray(g, np.float64)

    bounds = [(np.log(1e-3), np.log(1e3))] * dim + [(np.log(1e-3), np.log(1e3))]
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(4):
        theta0 = rng.uniform(np.log(0.1), np.log(5.0), size=dim + 1)
        result = spopt.minimize(fun, theta0, jac=True, method="L-BFGS-B", bounds=bounds)
        if best is None or result.fun < best.fun:
            best = result
    length_scales = [float(v) for v in np.exp(best.x[:dim])]
    variance = float(np.exp(best.x[dim]))
    return {**spec, "kernel": "rbf", "length_scales": length_scales, "variance": variance, "noise": noise}


# input-space bounds order for the DoE tools: conditions then time.
BOUND_VARS: Tuple[str, ...] = INPUT_VARS + ("t",)


def doe_bald(spec, state_payload, bounds, batch_size: int, seed: int = 0, n_restarts: int = 8):
    """Batch BALD: choose a batch B (``batch_size`` x INPUT_DIM) in the box ``bounds``
    (list of ``(lo, hi)`` per input dim) maximising the mutual information between the
    observations and ``f``, ``I(y_B; f) = 1/2 logdet(I + noise^-1 Sigma_BB)`` where
    ``Sigma_BB`` is the GP posterior covariance at B (prior ``K_BB`` if no data).
    Returns ``(B, eig_bits)``."""
    import jax
    import scipy.optimize as spopt
    from .kernels import gram

    gp = make_gp(spec)
    state = state_from_json(state_payload)
    k, noise = gp.kernel, gp.noise
    has_data = bool(state.X_flat.shape[0] > 0)
    dim = len(bounds)
    lo = jnp.asarray([b[0] for b in bounds], jnp.float32)
    hi = jnp.asarray([b[1] for b in bounds], jnp.float32)
    sp_bounds = [bounds[d] for _ in range(batch_size) for d in range(dim)]

    @jax.jit
    def neg_and_grad(b_flat):
        def obj(bf):
            B = bf.reshape(batch_size, dim)
            K_BB = gram(k, B, B)
            if has_data:
                K_tB = gram(k, state.X_flat, B)
                vB = jax.scipy.linalg.solve_triangular(state.L, K_tB, lower=True)
                S_BB = K_BB - vB.T @ vB
            else:
                S_BB = K_BB
            M = jnp.eye(batch_size) + (1.0 / noise) * S_BB
            return -0.5 * jnp.linalg.slogdet(M)[1]
        return jax.value_and_grad(obj)(b_flat)

    def fun(x):
        v, g = neg_and_grad(jnp.asarray(x, jnp.float32))
        return float(v), np.asarray(g, np.float64)

    best_x, best_f = None, np.inf
    for sub in jax.random.split(jax.random.PRNGKey(seed), n_restarts):
        b0 = jax.random.uniform(sub, (batch_size, dim), minval=lo, maxval=hi).reshape(-1)
        res = spopt.minimize(fun, np.asarray(b0, np.float64), jac=True, method="L-BFGS-B", bounds=sp_bounds)
        if res.fun < best_f:
            best_x, best_f = res.x, res.fun
    return np.asarray(best_x, np.float64).reshape(batch_size, dim), -float(best_f)


def discriminatory_doe(spec, state_payload, grid, threshold, bounds, batch_size: int, seed: int = 0):
    """Discriminative DoE: choose a batch best resolving the sign pattern
    ``[f(x) > threshold]`` over ``grid`` (see doe.doe.discriminative). Returns ``(B, eig_bits)``."""
    import jax
    from doe.doe.discriminative import optimize_batch

    gp = make_gp(spec)
    state = state_from_json(state_payload)
    X_grid = jnp.asarray(np.asarray(grid, dtype=np.float64), jnp.float32)
    B, eig = optimize_batch(gp, state, X_grid, float(threshold), batch_size, list(bounds),
                            jax.random.PRNGKey(seed))
    return np.asarray(B, dtype=np.float64), float(eig)


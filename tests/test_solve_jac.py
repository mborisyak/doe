"""
Tests for ODEModel.solve_jac — verifies shapes, finiteness, and correctness of
dX(t)/d(params) via comparison with central finite differences and jax.jacfwd.
"""
import json
import os
from collections import namedtuple

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import doe
from doe.common import Conditions
from doe.common.custom import CustomODESystem

ROOT = os.path.dirname(os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Minimal hand-written model (3 states, 3 params) — same as in test_model_base
# ---------------------------------------------------------------------------

Parameters = namedtuple('Parameters', ['q', 'K_A', 'K_B'])


class SimpleEnzyme(doe.common.ODEModel):
    def initial_state(self, conditions):
        Ac, Bc, Ec = doe.common.get_initial_concentrations(conditions)
        return jnp.stack([Ac, Bc, Ec], axis=-1)

    def rhs(self, state, conditions, parameters):
        A, B, E = state[..., 0], state[..., 1], state[..., 2]
        rate = E * parameters.q * A / (parameters.K_A + A) * B / (B + parameters.K_B)
        return jnp.stack([-rate, -rate, jnp.zeros_like(rate)], axis=-1)

    def observables(self, state, parameters):
        return state[..., 0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_trajectory(model, conditions, timestamps, parameters):
    """Run the IVP and return the full state trajectory (T, n), no observables."""
    from scipy.integrate import solve_ivp

    initial = np.array(model.initial_state(conditions), dtype=np.float64)
    rhs = lambda t, s: np.array(
        model.rhs(jnp.array(s, dtype=jnp.float32), conditions, parameters),
        dtype=np.float64,
    )
    T = float(np.max(timestamps))
    sol = solve_ivp(rhs, t_span=(0.0, T), t_eval=np.array(timestamps), y0=initial, method='LSODA')
    return sol.y.T  # (T, n)


def _fd_sensitivity(model, conditions, timestamps, parameters, eps=1e-3):
    """Central-difference dX(t)/d(param_i) for each param, shape (T, n, m)."""
    fields = parameters._fields
    base = _raw_trajectory(model, conditions, timestamps, parameters)
    T, n = base.shape
    result = np.zeros((T, n, len(fields)))

    for i, name in enumerate(fields):
        vals = parameters._asdict()

        vals[name] = float(getattr(parameters, name)) + eps
        traj_p = _raw_trajectory(model, conditions, timestamps, type(parameters)(**vals))

        vals[name] = float(getattr(parameters, name)) - eps
        traj_m = _raw_trajectory(model, conditions, timestamps, type(parameters)(**vals))

        result[:, :, i] = (traj_p - traj_m) / (2 * eps)

    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_model():
    return SimpleEnzyme()


@pytest.fixture
def custom_model():
    with open(os.path.join(ROOT, 'data', 'models', 'simple.json')) as f:
        spec = json.load(f)
    return CustomODESystem(spec)


CONDITIONS = Conditions(A=2.0, B=2.0, E=2.0, temperature=37.0)
TIMESTAMPS = jnp.array([1.0, 3.0, 5.0, 7.0, 9.0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_solve_jac_shape(simple_model):
    params = Parameters(q=500.0, K_A=0.5, K_B=0.5)
    trajectory, sensitivity = simple_model.solve_jac_scipy(CONDITIONS, TIMESTAMPS, params)

    T, n, m = len(TIMESTAMPS), 3, 3
    assert trajectory.shape == (T, n), f"trajectory: expected ({T},{n}), got {trajectory.shape}"
    assert sensitivity.shape == (T, n, m), f"sensitivity: expected ({T},{n},{m}), got {sensitivity.shape}"


def test_solve_jac_finite(simple_model):
    params = Parameters(q=500.0, K_A=0.5, K_B=0.5)
    trajectory, sensitivity = simple_model.solve_jac_scipy(CONDITIONS, TIMESTAMPS, params)

    assert np.all(np.isfinite(np.array(trajectory))), "trajectory contains non-finite values"
    assert np.all(np.isfinite(np.array(sensitivity))), "sensitivity contains non-finite values"


def test_solve_jac_trajectory_matches_scipy(simple_model):
    """State A in the augmented trajectory must match solve_scipy."""
    params = Parameters(q=500.0, K_A=0.5, K_B=0.5)
    trajectory, _ = simple_model.solve_jac_scipy(CONDITIONS, TIMESTAMPS, params)
    obs_scipy, _ = simple_model.solve_scipy(CONDITIONS, TIMESTAMPS, params)

    np.testing.assert_allclose(
        np.array(trajectory[:, 0]), np.array(obs_scipy),
        rtol=1e-2,
        err_msg="A-trajectory from solve_jac disagrees with solve_scipy",
    )


def test_solve_jac_vs_fd(simple_model):
    """solve_jac sensitivity must agree with central finite differences."""
    params = Parameters(q=500.0, K_A=0.5, K_B=0.5)
    _, sensitivity = simple_model.solve_jac_scipy(CONDITIONS, TIMESTAMPS, params)
    fd = _fd_sensitivity(simple_model, CONDITIONS, TIMESTAMPS, params, eps=1e-3)

    np.testing.assert_allclose(
        np.array(sensitivity), fd,
        rtol=1e-2, atol=1e-2,
        err_msg="sensitivity from solve_jac does not match finite differences",
    )


def test_solve_jac_custom_ode_shape(custom_model):
    """CustomODESystem (simple.json): check shapes."""
    params = custom_model.Parameters(q=500.0, K_A=0.5, K_B=0.5)
    trajectory, sensitivity = custom_model.solve_jac_scipy(CONDITIONS, TIMESTAMPS, params)

    n = len(custom_model.states)
    m = len(custom_model.parameters)
    T = len(TIMESTAMPS)
    assert trajectory.shape == (T, n)
    assert sensitivity.shape == (T, n, m)


def test_solve_jac_custom_ode_vs_fd(custom_model):
    """CustomODESystem (simple.json): sensitivity vs finite differences."""
    params = custom_model.Parameters(q=500.0, K_A=0.5, K_B=0.5)
    _, sensitivity = custom_model.solve_jac_scipy(CONDITIONS, TIMESTAMPS, params)
    fd = _fd_sensitivity(custom_model, CONDITIONS, TIMESTAMPS, params, eps=1e-3)

    np.testing.assert_allclose(
        np.array(sensitivity), fd,
        rtol=1e-2, atol=1e-2,
        err_msg="CustomODESystem sensitivity does not match finite differences",
    )


# ---------------------------------------------------------------------------
# jax.jacfwd helpers and benchmark
# ---------------------------------------------------------------------------

# Use a coarser Euler model for jacfwd — 1e6 steps is too slow to JIT/trace.
EULER_N = 10_000


def _make_jacfwd_fn(model, conditions, timestamps, parameters):
    """
    Build and JIT-compile a function that computes dX(t)/d(params) via
    jax.jacfwd on the Euler trajectory.  Parameters are passed as a flat
    array so JAX sees a plain R^m input.
    """
    def traj_fn(params_flat):
        ps = type(parameters)(*params_flat)
        traj, _ = model.trajectory_euler(conditions, timestamps, ps)
        return traj  # (T, n_states)

    return jax.jit(jax.jacfwd(traj_fn))


def _params_flat(parameters):
    return jnp.stack([jnp.asarray(getattr(parameters, k), dtype=jnp.float32)
                      for k in parameters._fields])


def test_jacfwd_shape():
    """jax.jacfwd on Euler trajectory produces correct (T, n, m) shape."""
    model = SimpleEnzyme(n=EULER_N)
    params = Parameters(q=500.0, K_A=0.5, K_B=0.5)

    jac_fn = _make_jacfwd_fn(model, CONDITIONS, TIMESTAMPS, params)
    sensitivity = jac_fn(_params_flat(params))

    T, n, m = len(TIMESTAMPS), 3, 3
    assert sensitivity.shape == (T, n, m)


def test_jacfwd_vs_fd():
    """jax.jacfwd sensitivity matches finite differences (Euler solver)."""
    model = SimpleEnzyme(n=EULER_N)
    params = Parameters(q=500.0, K_A=0.5, K_B=0.5)

    jac_fn = _make_jacfwd_fn(model, CONDITIONS, TIMESTAMPS, params)
    sensitivity = jac_fn(_params_flat(params))
    fd = _fd_sensitivity(model, CONDITIONS, TIMESTAMPS, params, eps=1e-3)

    np.testing.assert_allclose(
        np.array(sensitivity), fd,
        rtol=0.05, atol=0.05,
        err_msg="jacfwd sensitivity does not match finite differences",
    )


def test_solve_jac_vs_jacfwd():
    """
    End-to-end comparison and wall-clock benchmark of solve_jac (sensitivity
    equations + LSODA) vs jax.jacfwd (forward-mode AD through Euler scan).
    Both use the same model with n=EULER_N Euler steps for a fair comparison.
    """
    import time

    N_RUNS = 100
    model = SimpleEnzyme(n=EULER_N)
    params = Parameters(q=500.0, K_A=0.5, K_B=0.5)

    # --- solve_jac ---
    _, sens_jac = model.solve_jac_scipy(CONDITIONS, TIMESTAMPS, params)   # warm-up
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        _, sens_jac = model.solve_jac_scipy(CONDITIONS, TIMESTAMPS, params)
    time_jac = (time.perf_counter() - t0) / N_RUNS

    # --- jax.jacfwd ---
    jac_fn = _make_jacfwd_fn(model, CONDITIONS, TIMESTAMPS, params)
    p_flat = _params_flat(params)
    sens_fwd = jax.block_until_ready(jac_fn(p_flat))                # JIT warm-up
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        sens_fwd = jax.block_until_ready(jac_fn(p_flat))
    time_fwd = (time.perf_counter() - t0) / N_RUNS

    # --- solve_jac_euler (sensitivity eqs + Euler scan, JIT) ---
    euler_fn = jax.jit(lambda p: model.solve_jac_euler(CONDITIONS, TIMESTAMPS, p))
    _, sens_euler = jax.block_until_ready(euler_fn(params))              # JIT warm-up
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        _, sens_euler = jax.block_until_ready(euler_fn(params))
    time_euler = (time.perf_counter() - t0) / N_RUNS

    print(f"\nsolve_jac        (sensitivity eqs + LSODA):      {time_jac   * 1e3:7.1f} ms")
    print(f"solve_jac_euler  (sensitivity eqs + Euler JIT):  {time_euler * 1e3:7.1f} ms")
    print(f"jax.jacfwd       (forward AD   + Euler JIT):     {time_fwd   * 1e3:7.1f} ms")
    print(f"Speedup euler  vs LSODA: {time_jac / time_euler:.1f}x")
    print(f"Speedup jacfwd vs LSODA: {time_jac / time_fwd:.1f}x")

    assert sens_jac.shape == sens_fwd.shape == sens_euler.shape

    # solve_jac_euler: same equations, different solver → should be close
    np.testing.assert_allclose(
        np.array(sens_jac), np.array(sens_euler),
        rtol=0.10, atol=0.05,
        err_msg="solve_jac and solve_jac_euler give substantially different sensitivities",
    )
    # jacfwd: different formulation → generous tolerance
    np.testing.assert_allclose(
        np.array(sens_jac), np.array(sens_fwd),
        rtol=0.10, atol=0.05,
        err_msg="solve_jac and jacfwd give substantially different sensitivities",
    )
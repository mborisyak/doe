"""
Tests for ODEModel.observable_jac_scipy — verifies shape, finiteness, and
correctness against jax.jacfwd applied directly to the Euler solver.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from collections import namedtuple

import doe
import doe.common
from doe.common import Conditions

Parameters = namedtuple('Parameters', ['q', 'K_A', 'K_B'])

CONDITIONS  = Conditions(A=2.0, B=2.0, E=2.0, temperature=37.0)
TIMESTAMPS  = jnp.array([1.0, 3.0, 5.0, 7.0, 9.0])
PARAMS      = Parameters(q=500.0, K_A=0.5, K_B=0.5)
EULER_N     = 1_000  # coarse enough to JIT quickly


class SimpleEnzyme(doe.common.ODEModel):
    def initial_state(self, conditions):
        Ac, Bc, Ec = doe.common.get_initial_concentrations(conditions)
        return jnp.stack([Ac, Bc, Ec], axis=-1)

    def rhs(self, state, conditions, parameters):
        A, B, E = state[..., 0], state[..., 1], state[..., 2]
        rate = E * parameters.q * A / (parameters.K_A + A) * B / (B + parameters.K_B)
        return jnp.stack([-rate, -rate, jnp.zeros_like(rate)], axis=-1)

    def observables(self, state, parameters):
        return state[..., :2]  # (T, 2) — two observables so the Jacobian is non-trivial


@pytest.fixture(scope='module')
def model():
    return SimpleEnzyme(n=EULER_N)


def _params_flat(parameters):
    return jnp.stack([jnp.asarray(v, dtype=jnp.float32) for v in parameters])


def _jacfwd_euler(model, conditions, timestamps, parameters):
    """Reference: jax.jacfwd on model.solve (Euler)."""
    def solve_fn(dp):
        ps = type(parameters)(*dp)
        obs, _ = model.solve_euler(conditions, timestamps, ps)
        return obs

    return jax.jit(jax.jacfwd(solve_fn))(_params_flat(parameters))


def _fd_observable_jac(model, conditions, timestamps, parameters, eps=1e-3):
    """Ground truth: central finite differences of observables w.r.t. each parameter."""
    from scipy.integrate import solve_ivp

    def run(params):
        initial = np.array(model.initial_state(conditions), dtype=np.float64)
        rhs = lambda t, s: np.array(
            model.rhs(jnp.array(s, dtype=jnp.float32), conditions, params), dtype=np.float64
        )
        T = float(np.max(timestamps))
        sol = solve_ivp(rhs, t_span=(0.0, T), t_eval=np.array(timestamps), y0=initial, method='LSODA')
        states = jnp.array(sol.y.T, dtype=jnp.float32)
        return np.array(model.observables(states, params))  # (T, k)

    fields = parameters._fields
    base = run(parameters)
    T, k = base.shape
    m = len(fields)
    jac = np.zeros((T, k, m))

    for i, name in enumerate(fields):
        vals = parameters._asdict()
        vals[name] = float(getattr(parameters, name)) + eps
        obs_p = run(type(parameters)(**vals))
        vals[name] = float(getattr(parameters, name)) - eps
        obs_m = run(type(parameters)(**vals))
        jac[:, :, i] = (obs_p - obs_m) / (2 * eps)

    return jac


# ---------------------------------------------------------------------------

def test_observable_jac_scipy_shape(model):
    obs, jac = model.observable_jac_scipy(CONDITIONS, TIMESTAMPS, PARAMS)

    T, k, m = len(TIMESTAMPS), 2, len(PARAMS)
    assert obs.shape == (T, k),    f"obs: expected {(T, k)}, got {obs.shape}"
    assert jac.shape == (T, k, m), f"jac: expected {(T, k, m)}, got {jac.shape}"


def test_observable_jac_scipy_finite(model):
    obs, jac = model.observable_jac_scipy(CONDITIONS, TIMESTAMPS, PARAMS)

    assert np.all(np.isfinite(np.array(obs))),  "obs contains non-finite values"
    assert np.all(np.isfinite(np.array(jac))), "jac contains non-finite values"


def test_observable_jac_scipy_obs_matches_solve(model):
    """Observables returned by observable_jac_scipy must match model.solve_scipy."""
    obs_jac, _ = model.observable_jac_scipy(CONDITIONS, TIMESTAMPS, PARAMS)
    obs_solve = model.solve_scipy(CONDITIONS, TIMESTAMPS, PARAMS)

    np.testing.assert_allclose(
        np.array(obs_jac), np.array(obs_solve),
        rtol=1e-3, atol=1e-3,
        err_msg="observable_jac_scipy obs disagrees with solve_scipy",
    )


def test_observable_jac_scipy_vs_fd(model):
    """Jacobian from observable_jac_scipy must agree with central finite differences."""
    _, jac_scipy = model.observable_jac_scipy(CONDITIONS, TIMESTAMPS, PARAMS)
    jac_fd = _fd_observable_jac(model, CONDITIONS, TIMESTAMPS, PARAMS, eps=1e-3)

    np.testing.assert_allclose(
        np.array(jac_scipy), jac_fd,
        rtol=1e-2, atol=1e-2,
        err_msg="observable_jac_scipy Jacobian disagrees with finite differences",
    )


def test_observable_jac_scipy_vs_jacfwd(model):
    """Jacobian from observable_jac_scipy must agree with jax.jacfwd on Euler."""
    _, jac_scipy = model.observable_jac_scipy(CONDITIONS, TIMESTAMPS, PARAMS)
    jac_fwd = _jacfwd_euler(model, CONDITIONS, TIMESTAMPS, PARAMS)

    np.testing.assert_allclose(
        np.array(jac_scipy), np.array(jac_fwd),
        rtol=1.0e-3, atol=1.0e-3,
        err_msg="observable_jac_scipy Jacobian disagrees with jax.jacfwd on Euler solver",
    )
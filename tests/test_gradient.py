"""
Tests for ODEModel.gradient_scipy.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from collections import namedtuple
from scipy.integrate import solve_ivp

import doe
import doe.common
from doe.common import Conditions

Parameters = namedtuple('Parameters', ['q', 'K_A', 'K_B'])

CONDITIONS = Conditions(A=2.0, B=2.0, E=2.0, temperature=37.0)
TIMESTAMPS = jnp.array([1.0, 3.0, 5.0, 7.0, 9.0])
PARAMS     = Parameters(q=500.0, K_A=0.5, K_B=0.5)
EULER_N    = 10_000


class SimpleEnzyme(doe.common.ODEModel):
    def initial_state(self, conditions):
        Ac, Bc, Ec = doe.common.get_initial_concentrations(conditions)
        return jnp.stack([Ac, Bc, Ec], axis=-1)

    def rhs(self, state, conditions, parameters):
        A, B, E = state[..., 0], state[..., 1], state[..., 2]
        rate = E * parameters.q * A / (parameters.K_A + A) * B / (B + parameters.K_B)
        return jnp.stack([-rate, -rate, jnp.zeros_like(rate)], axis=-1)

    def observables(self, state, parameters):
        return state[..., :2]


@pytest.fixture(scope='module')
def model():
    return SimpleEnzyme(n=EULER_N)


TARGETS = jnp.ones((len(TIMESTAMPS), 2)) * 0.5

def loss(trajectory, targets):
    return jnp.sum((trajectory[..., :2] - targets) ** 2)


def _fd_gradient(model, conditions, timestamps, parameters, targets, eps=1e-3):
    """Central finite differences of loss w.r.t. each parameter via LSODA."""
    def run_loss(params):
        initial = np.array(model.initial_state(conditions), dtype=np.float64)
        rhs = lambda t, s: np.array(
            model.rhs(jnp.array(s, dtype=jnp.float32), conditions, params), dtype=np.float64
        )
        T = float(np.max(timestamps))
        sol = solve_ivp(rhs, t_span=(0.0, T), t_eval=np.array(timestamps), y0=initial, method='LSODA')
        traj = jnp.array(sol.y.T, dtype=jnp.float32)
        return float(loss(traj, targets))

    grad = {}
    for name in parameters._fields:
        vals = parameters._asdict()
        vals[name] = float(getattr(parameters, name)) + eps
        lp = run_loss(type(parameters)(**vals))
        vals[name] = float(getattr(parameters, name)) - eps
        lm = run_loss(type(parameters)(**vals))
        grad[name] = (lp - lm) / (2 * eps)

    return type(parameters)(**grad)


def _fd_state_jac(model, conditions, timestamps, parameters, eps=1e-3):
    """FD d(state)/d(params) via LSODA, shape (T, n_states, m)."""
    def run(params):
        initial = np.array(model.initial_state(conditions), dtype=np.float64)
        rhs = lambda t, s: np.array(
            model.rhs(jnp.array(s, dtype=jnp.float32), conditions, params), dtype=np.float64
        )
        T = float(np.max(timestamps))
        sol = solve_ivp(rhs, t_span=(0.0, T), t_eval=np.array(timestamps), y0=initial, method='LSODA')
        return sol.y.T  # (T, n_states)

    fields = parameters._fields
    base = run(parameters)
    T, n = base.shape
    jac = np.zeros((T, n, len(fields)))
    for i, name in enumerate(fields):
        vals = parameters._asdict()
        vals[name] = float(getattr(parameters, name)) + eps
        yp = run(type(parameters)(**vals))
        vals[name] = float(getattr(parameters, name)) - eps
        ym = run(type(parameters)(**vals))
        jac[:, :, i] = (yp - ym) / (2 * eps)
    return jac


def _jacfwd_single_scan(model, conditions, timestamps, parameters):
    """jax.jacfwd d(state)/d(params) via single Euler scan, shape (T, n_states, m)."""
    def traj_fn(dp):
        ps = type(parameters)(*(dp[..., i] for i, _ in enumerate(parameters)))
        initial = model.initial_state(conditions)
        T = jnp.max(timestamps)
        dt = T / model.n

        def step(state, _):
            ds_dt = model.rhs(state, conditions, ps)
            updated = state + dt * ds_dt
            updated = jnp.clip(updated, 0)
            return updated, updated

        _, full_traj = jax.lax.scan(step, init=initial, length=model.n)
        index = jnp.clip(jnp.ceil(timestamps / dt).astype(int), 0, model.n - 1)
        return full_traj[index]

    dp = jnp.stack([jnp.asarray(v, dtype=jnp.float32) for v in parameters])
    return jax.jit(jax.jacfwd(traj_fn))(dp)  # (T, n_states, m)


def _jacrev_single_scan(model, conditions, timestamps, parameters):
    """jax.jacrev d(state)/d(params) via single Euler scan, shape (T, n_states, m)."""
    def traj_fn(dp):
        ps = type(parameters)(*(dp[..., i] for i, _ in enumerate(parameters)))
        initial = model.initial_state(conditions)
        T = jnp.max(timestamps)
        dt = T / model.n

        def step(state, _):
            ds_dt = model.rhs(state, conditions, ps)
            updated = state + dt * ds_dt
            updated = jnp.clip(updated, 0)
            return updated, updated

        _, full_traj = jax.lax.scan(step, init=initial, length=model.n)
        index = jnp.clip(jnp.ceil(timestamps / dt).astype(int), 0, model.n - 1)
        return full_traj[index]

    dp = jnp.stack([jnp.asarray(v, dtype=jnp.float32) for v in parameters])
    return jax.jit(jax.jacrev(traj_fn))(dp)  # (T, n_states, m)


def _grad_model_solve(model, conditions, timestamps, parameters, targets):
    """jax.grad through model.solve (double Euler scan + observables)."""
    def total_loss(dp):
        ps = type(parameters)(*(dp[..., i] for i, _ in enumerate(parameters)))
        obs, _ = model.solve_euler(conditions, timestamps, ps)
        return loss(obs, targets)
    dp = jnp.stack([jnp.asarray(v, dtype=jnp.float32) for v in parameters])
    return jax.jit(jax.grad(total_loss))(dp)


def _make_scan_loss(model, conditions, timestamps, parameters, targets):
    """Scalar loss as a function of flat parameter vector, via single Euler scan."""
    def total_loss(dp):
        ps = type(parameters)(*(dp[..., i] for i, _ in enumerate(parameters)))
        initial = model.initial_state(conditions)
        T = jnp.max(timestamps)
        dt = T / model.n

        def step(state, _):
            ds_dt = model.rhs(state, conditions, ps)
            updated = state + dt * ds_dt
            updated = jnp.clip(updated, 0)
            return updated, updated

        _, full_traj = jax.lax.scan(step, init=initial, length=model.n)
        index = jnp.clip(jnp.ceil(timestamps / dt).astype(int), 0, model.n - 1)
        traj = full_traj[index]
        return loss(traj, targets)

    return total_loss


def _grad_custom_scan(model, conditions, timestamps, parameters, targets):
    """jax.grad (reverse-mode) through a single Euler scan."""
    f  = _make_scan_loss(model, conditions, timestamps, parameters, targets)
    dp = jnp.stack([jnp.asarray(v, dtype=jnp.float32) for v in parameters])
    return jax.jit(jax.grad(f))(dp), jax.jit(jax.jacfwd(f))(dp)


# ---------------------------------------------------------------------------

def test_gradient_scipy_returns_namedtuple(model):
    _, g = model.value_gradient_scipy(loss, CONDITIONS, TIMESTAMPS, PARAMS, arguments=TARGETS)
    assert type(g) is Parameters
    assert g._fields == PARAMS._fields


def test_gradient_scipy_finite(model):
    _, g = model.value_gradient_scipy(loss, CONDITIONS, TIMESTAMPS, PARAMS, arguments=TARGETS)
    assert all(np.isfinite(float(v)) for v in g), f"gradient contains non-finite values: {g}"


def test_gradient_scipy_vs_fd(model):
    """value_gradient_scipy must agree with central finite differences."""
    _, g = model.value_gradient_scipy(loss, CONDITIONS, TIMESTAMPS, PARAMS, arguments=TARGETS)
    g_fd = _fd_gradient(model, CONDITIONS, TIMESTAMPS, PARAMS, TARGETS, eps=1e-3)

    for name in Parameters._fields:
        np.testing.assert_allclose(
            float(getattr(g, name)), float(getattr(g_fd, name)),
            rtol=1e-2, atol=1e-2,
            err_msg=f"value_gradient_scipy disagrees with FD for parameter '{name}'",
        )


def test_gradient_scipy_vs_euler_grad(plot_root):
    """Compare state sensitivities and loss gradients across four methods each."""
    import matplotlib.pyplot as plt

    model  = SimpleEnzyme(n=1_000)
    fields = Parameters._fields
    ts     = np.array(TIMESTAMPS)

    # assert clip never activates during the full Euler trajectory
    initial = model.initial_state(CONDITIONS)
    T_max = float(jnp.max(TIMESTAMPS))
    dt = T_max / model.n
    def _step(state, _):
        return jnp.clip(state + dt * model.rhs(state, CONDITIONS, PARAMS), 0), state + dt * model.rhs(state, CONDITIONS, PARAMS)
    _, unclipped = jax.lax.scan(_step, init=initial, length=model.n)
    assert float(jnp.min(unclipped)) > 0, f"clip is active: min state = {float(jnp.min(unclipped))}"

    # ---- state sensitivities: d(state)/d(params), shape (T, n_states, m) ----
    traj_scipy, S_scipy = model.solve_jac_scipy(CONDITIONS, TIMESTAMPS, PARAMS)
    traj_euler, S_euler = model.solve_jac_euler(CONDITIONS, TIMESTAMPS, PARAMS)
    S_jacfwd            = np.array(_jacfwd_single_scan(model, CONDITIONS, TIMESTAMPS, PARAMS))
    S_jacrev            = np.array(_jacrev_single_scan(model, CONDITIONS, TIMESTAMPS, PARAMS))
    S_fd                = _fd_state_jac(model, CONDITIONS, TIMESTAMPS, PARAMS)

    # ---- gradients of loss w.r.t. params ----
    _, g_scipy       = model.value_gradient_scipy(loss, CONDITIONS, TIMESTAMPS, PARAMS, arguments=TARGETS)
    g_model_solve        = _grad_model_solve(model, CONDITIONS, TIMESTAMPS, PARAMS, TARGETS)
    g_jacrev, g_jacfwd  = _grad_custom_scan(model, CONDITIONS, TIMESTAMPS, PARAMS, TARGETS)
    g_fd                 = _fd_gradient(model, CONDITIONS, TIMESTAMPS, PARAMS, TARGETS)

    print("\n--- state sensitivity [t=0, state=0, all params] ---")
    print(f"S_scipy  : {np.array(S_scipy [0, 0, :])}")
    print(f"S_euler  : {np.array(S_euler [0, 0, :])}")
    print(f"S_jacfwd : {S_jacfwd[0, 0, :]}")
    print(f"S_jacrev : {S_jacrev[0, 0, :]}")
    print(f"S_fd     : {S_fd    [0, 0, :]}")
    g_scipy_arr = np.array(jnp.stack(list(g_scipy)))
    g_fd_arr    = np.array(jnp.stack(list(g_fd)))
    g_ms_arr    = np.array(g_model_solve)
    g_jr_arr    = np.array(g_jacrev)
    g_jf_arr    = np.array(g_jacfwd)

    print("\n--- gradient ---")
    header = f"{'param':<8}" + "".join(f"{n:>14}" for n in ['scipy', 'model.solve', 'jacrev', 'jacfwd', 'FD'])
    print(header)
    for i, name in enumerate(fields):
        row = f"{name:<8}" + "".join(f"{v:>14.6f}" for v in [g_scipy_arr[i], g_ms_arr[i], g_jr_arr[i], g_jf_arr[i], g_fd_arr[i]])
        print(row)
    print("\n--- jacrev / jacfwd ratio ---")
    print(g_jr_arr / g_jf_arr)

    # manual gradient: cotangent · S, using S from each Jacobian method
    # if S is correct, manual grad should match jacfwd regardless of which S we use
    traj_euler_selected = np.array(traj_euler)   # (T, n_states) at measurement times
    ct = np.zeros_like(traj_euler_selected)       # (T, n_states)
    ct[:, :2] = 2 * (traj_euler_selected[:, :2] - np.array(TARGETS))

    g_manual_jacfwd = np.einsum('ti,tij->j', ct, S_jacfwd)
    g_manual_jacrev = np.einsum('ti,tij->j', ct, S_jacrev)

    print("\n--- manual gradient (ct · S) ---")
    print(f"ct · S_jacfwd : {g_manual_jacfwd}")
    print(f"ct · S_jacrev : {g_manual_jacrev}")
    print(f"g_jacfwd      : {g_jf_arr}")
    print(f"g_jacrev      : {g_jr_arr}")

    # vary n: does the ratio depend on it?
    print("\n--- S_jacrev / S_jacfwd (per timestamp, state 0) ---")
    for t_idx in range(len(TIMESTAMPS)):
        ratio = S_jacrev[t_idx, 0, :] / S_jacfwd[t_idx, 0, :]
        print(f"  t={float(TIMESTAMPS[t_idx]):.0f}  ratio={ratio}")

    print("\n--- jacrev/jacfwd ratio vs n ---")
    for n_test in [100, 500, 1000, 2000]:
        m_test = SimpleEnzyme(n=n_test)
        f_test = _make_scan_loss(m_test, CONDITIONS, TIMESTAMPS, PARAMS, TARGETS)
        dp     = jnp.stack([jnp.asarray(v, dtype=jnp.float32) for v in PARAMS])
        gr = jax.jit(jax.grad(f_test))(dp)
        gf = jax.jit(jax.jacfwd(f_test))(dp)
        print(f"  n={n_test:5d}  ratio={np.array(gr / gf)}")

    n_states = np.array(S_scipy).shape[1]
    m        = len(fields)
    traj_scipy_np = np.array(traj_scipy)
    traj_euler_np = np.array(traj_euler)

    # plot 1: col 0 = state trajectories, cols 1..m = sensitivities
    fig, axes = plt.subplots(n_states, 1 + m, figsize=(4 * (1 + m), 3 * n_states), sharex=True)
    for i in range(n_states):
        # trajectory column
        ax = axes[i, 0]
        ax.plot(ts, traj_scipy_np[:, i], 'o-',  label='scipy')
        ax.plot(ts, traj_euler_np[:, i], 's--', label='euler')
        ax.set_title(f'x{i}(t)')
        ax.set_xlabel('t')
        if i == 0:
            ax.legend(fontsize=7)
        # sensitivity columns
        for j, fname in enumerate(fields):
            ax = axes[i, 1 + j]
            ax.plot(ts, S_fd    [:, i, j],           '^--', label='FD',     color='k')
            ax.plot(ts, np.array(S_scipy[:, i, j]),  'o-',  label='scipy')
            ax.plot(ts, np.array(S_euler[:, i, j]),  's--', label='euler')
            ax.plot(ts, S_jacfwd[:, i, j],           'D:',  label='jacfwd')
            ax.plot(ts, S_jacrev[:, i, j],           'v:',  label='jacrev')
            ax.axhline(0, color='k', lw=0.5, ls='--')
            ax.set_title(f'dx{i}/d({fname})')
            ax.set_xlabel('t')
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{plot_root}/sensitivity.png', dpi=120)
    plt.close(fig)

    # plot 2: gradients
    g_scipy_arr = np.array(jnp.stack(list(g_scipy)))
    g_ms_arr    = np.array(g_model_solve)
    g_jr_arr    = np.array(g_jacrev)
    g_jf_arr    = np.array(g_jacfwd)
    g_fd_arr    = np.array(jnp.stack(list(g_fd)))

    x, w = np.arange(m), 0.15
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 2 * w, g_fd_arr,    w, label='FD',          color='k')
    ax.bar(x - 1 * w, g_scipy_arr, w, label='scipy')
    ax.bar(x + 0 * w, g_ms_arr,    w, label='model.solve')
    ax.bar(x + 1 * w, g_jr_arr,    w, label='scan (rev)')
    ax.bar(x + 2 * w, g_jf_arr,    w, label='scan (fwd)')
    ax.set_xticks(x)
    ax.set_xticklabels(fields)
    ax.set_ylabel('gradient')
    ax.set_title('Gradient comparison')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{plot_root}/gradient.png', dpi=120)
    plt.close(fig)

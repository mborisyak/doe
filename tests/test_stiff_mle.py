"""
Fit a stiff full mass-action system to data generated from the reduced
Michaelis–Menten equation.

True data model (MM quasi-steady-state):
    dA/dt = -A / (A + K),   K = 1.0,  Vmax = 1.0

Fit model — full mass-action mechanism, stiff when k_on, k_off >> k_cat:
    dA/dt  = -k_on·A·E + k_off·AE
    dE/dt  = -k_on·A·E + (k_off + k_cat)·AE
    dAE/dt =  k_on·A·E - (k_off + k_cat)·AE

Under QSS: K_M = (k_off + k_cat) / k_on,  Vmax = k_cat · E_total.
Parameter ranges are constrained to the stiff regime (k_on, k_off ≥ 500),
so Euler will fail and auto mode must fall back to scipy/LSODA.
"""
import os
from collections import namedtuple

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import jax.numpy as jnp

import doe
from doe.common import ODEModel, Conditions
from doe.inference.gradient import (
  maximum_likelihood_estimate,
  maximum_likelihood_estimate_euler,
)


# ---------------------------------------------------------------------------
# Mass-action model
# ---------------------------------------------------------------------------

MAParameters = namedtuple('MAParameters', ['k_on', 'k_off', 'k_cat'])


class MassActionMM(ODEModel):
  """Full mass-action Michaelis–Menten.  State: [A, E, AE].  Observable: A."""

  def initial_state(self, conditions):
    return jnp.array([conditions.A, conditions.E, 0.0])

  def rhs(self, state, conditions, parameters):
    A, E, AE = state[0], state[1], state[2]
    forward   = parameters.k_on  * A * E
    reverse   = parameters.k_off * AE
    catalysis = parameters.k_cat * AE
    return jnp.array([
      -forward + reverse,
      -forward + reverse + catalysis,
       forward - reverse - catalysis,
    ])

  def observables(self, state, parameters):
    return state[..., 0]  # substrate A

  def parameter_ranges(self):
    # stiff regime: both fast rates ≥ 500
    return MAParameters(k_on=(10.0, 100.0), k_off=(10.0, 100.0), k_cat=(0.1, 5.0))


# ---------------------------------------------------------------------------
# Data generation from the reduced MM ODE
# ---------------------------------------------------------------------------

def generate_mm_data(A0_list, E0=1.0, K=1.0, T=8.0, n_points=10,
                     noise_sigma=0.02, seed=42):
  """Integrate dA/dt = -A/(A+K) with scipy and add Gaussian noise."""
  rng = np.random.default_rng(seed)
  timestamps = np.linspace(T / n_points, T, n_points)

  conditions, measurements = {}, {}
  for i, A0 in enumerate(A0_list):
    sol = solve_ivp(
      lambda t, y: [-y[0] / (y[0] + K)],
      t_span=(0.0, T), t_eval=timestamps, y0=[A0], method='LSODA',
    )
    A_obs = np.clip(sol.y[0] + rng.normal(0, noise_sigma, size=sol.y[0].shape), 0, None)

    label = f'exp_{i + 1}'
    conditions[label]   = {'A': float(A0), 'B': 0.0, 'E': float(E0), 'temperature': 37.0}
    measurements[label] = {'timestamps': timestamps.tolist(), 'measurements': A_obs.tolist()}

  return conditions, measurements


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_stiff_mass_action_fit(plot_root):
  """
  Euler must fail (integration error too large in the stiff regime).
  Auto mode falls back to scipy/LSODA and recovers the effective MM parameters.
  """
  import matplotlib.pyplot as plt

  conditions, measurements = generate_mm_data(A0_list=[2.0, 4.0])
  model = MassActionMM()
  parameter_ranges = model.parameter_ranges()

  # --- euler: expected to hit integration error ---
  euler_failed = False
  try:
    losses_euler, params_euler, preds_euler = maximum_likelihood_estimate_euler(
      model, conditions, measurements, parameter_ranges,
      iterations=256,
    )
  except AssertionError as exc:
    print(f'euler failed (expected): {exc}')
    euler_failed = True
    losses_euler = preds_euler = None

  # --- auto: should fall back to scipy ---
  losses_auto, params_auto, preds_auto = maximum_likelihood_estimate(
    model, conditions, measurements, parameter_ranges,
    iterations=64, mode='auto',
  )

  K_M_fit   = (float(params_auto.k_off) + float(params_auto.k_cat)) / float(params_auto.k_on)
  Vmax_fit  = float(params_auto.k_cat) * 1.0  # E_total = 1

  print(f'auto final loss : {losses_auto[-1]:.6f}')
  print(f'fitted K_M      = {K_M_fit:.4f}  (true 1.0)')
  print(f'fitted Vmax     = {Vmax_fit:.4f}  (true 1.0)')
  print(f'k_on={float(params_auto.k_on):.1f}  k_off={float(params_auto.k_off):.1f}  k_cat={float(params_auto.k_cat):.4f}')

  # --- plot ---
  labels = list(measurements.keys())
  n = len(labels)
  fig, axes = plt.subplots(n + 1, 1, figsize=(8, 4 * (n + 1)))

  ax0 = axes[0]
  ax0.plot(losses_auto, color='tab:orange', label=f'auto/scipy  (final={losses_auto[-1]:.4g})')
  if not euler_failed:
    ax0.plot(losses_euler, color='tab:blue', label=f'euler  (final={losses_euler[-1]:.4g})')
  ax0.set_xlabel('optimizer step')
  ax0.set_ylabel('MSE loss')
  ax0.set_title('Learning curves' + ('  [euler: crashed — integration error]' if euler_failed else ''))
  ax0.legend()

  for i, label in enumerate(labels):
    ts = np.array(measurements[label]['timestamps'])
    ys = np.array(measurements[label]['measurements'])
    ax = axes[i + 1]
    ax.scatter(ts, ys, color='black', zorder=3, label='MM data (noisy)')
    ax.plot(ts, np.array(preds_auto[label]), color='tab:orange', marker='s', label='auto/scipy fit')
    if not euler_failed:
      ax.plot(ts, np.array(preds_euler[label]), color='tab:blue', marker='o', label='euler fit')
    ax.set_title(f'{label}  (A₀={conditions[label]["A"]:.1f})')
    ax.set_xlabel('time')
    ax.set_ylabel('substrate A')
    ax.legend()

  fig.suptitle(
    f'Mass-action fit to MM data\n'
    f'fitted: K_M≈{K_M_fit:.3f} (true 1.0),  Vmax≈{Vmax_fit:.3f} (true 1.0)',
    fontsize=10,
  )
  fig.tight_layout()
  fig.savefig(os.path.join(plot_root, 'stiff_mass_action.png'), dpi=120)
  plt.close(fig)

  assert losses_auto[-1] < 0.05, f'auto fit loss too high: {losses_auto[-1]:.4g}'
  assert abs(K_M_fit  - 1.0) < 0.3, f'K_M far from truth: {K_M_fit:.3f}'
  assert abs(Vmax_fit - 1.0) < 0.3, f'Vmax far from truth: {Vmax_fit:.3f}'

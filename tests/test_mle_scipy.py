"""
Tests for maximum_likelihood_estimate_scipy and comparison against the euler variant.
"""
import os
import json

import numpy as np
import pytest

import jax.numpy as jnp

import doe
import doe.inference
from doe.common import Conditions
from doe.inference.gradient import (
  maximum_likelihood_estimate_euler,
  maximum_likelihood_estimate_scipy,
  maximum_likelihood_estimate,
)


@pytest.fixture(scope='module')
def simple_setup():
  root = os.path.dirname(os.path.dirname(__file__))

  with open(os.path.join(root, 'data', 'models', 'simple.json'), 'r') as f:
    model = doe.common.CustomODESystem(json.load(f))

  with open(os.path.join(root, 'data', 'experiments', 'example.json'), 'r') as f:
    conditions = json.load(f)

  with open(os.path.join(root, 'data', 'experiments', 'measurements.json'), 'r') as f:
    measurements = json.load(f)

  # use a small subset for speed
  labels = list(conditions.keys())[:2]
  conditions = {k: conditions[k] for k in labels}
  measurements = {k: measurements[k] for k in labels}

  return model, conditions, measurements


def test_scipy_mle_runs(simple_setup):
  """scipy MLE runs without error and returns finite loss / valid parameters."""
  model, conditions, measurements = simple_setup
  parameter_ranges = model.parameter_ranges()

  losses, parameters, predictions = maximum_likelihood_estimate_scipy(
    model, conditions, measurements, parameter_ranges, iterations=50
  )

  assert len(losses) > 0
  assert np.all(np.isfinite(losses)), 'losses contain non-finite values'
  assert np.all(np.isfinite(list(parameters))), 'parameters contain non-finite values'

  # parameters must stay within bounds
  for val, (low, high) in zip(parameters, parameter_ranges):
    assert float(low) <= float(val) <= float(high), \
      f'parameter {float(val):.4g} out of bounds [{float(low)}, {float(high)}]'

  # predictions exist for every condition
  assert set(predictions.keys()) == set(conditions.keys())
  for label, pred in predictions.items():
    assert np.all(np.isfinite(np.array(pred))), f'predictions for {label!r} are non-finite'


def test_scipy_mle_loss_decreases(simple_setup):
  """Loss should be non-increasing: final loss <= initial loss."""
  model, conditions, measurements = simple_setup
  parameter_ranges = model.parameter_ranges()

  losses, _, _ = maximum_likelihood_estimate_scipy(
    model, conditions, measurements, parameter_ranges, iterations=100
  )

  assert losses[-1] <= losses[0] + 1e-10, \
    f'final loss ({losses[-1]:.4g}) is higher than initial ({losses[0]:.4g})'


def test_scipy_vs_euler_convergence(simple_setup, plot_root):
  """Both methods should converge to a similar loss from the same starting point."""
  import matplotlib.pyplot as plt

  model, conditions, measurements = simple_setup
  parameter_ranges = model.parameter_ranges()

  losses_euler, params_euler, preds_euler = maximum_likelihood_estimate_euler(
    model, conditions, measurements, parameter_ranges, iterations=512
  )

  losses_scipy, params_scipy, preds_scipy = maximum_likelihood_estimate_scipy(
    model, conditions, measurements, parameter_ranges, iterations=200
  )
  print(losses_scipy)
  print(f'euler  final loss: {losses_euler[-1]:.6f}')
  print(f'scipy  final loss: {losses_scipy[-1]:.6f}')

  # --- plot ---
  labels = list(measurements.keys())
  n = len(labels)
  fig, axes = plt.subplots(n + 1, 1, figsize=(8, 4 * (n + 1)))

  # learning curves
  axes[0].plot(losses_euler, label=f'euler (final={losses_euler[-1]:.4g})')
  axes[0].plot(
    np.linspace(0, len(losses_euler) - 1, len(losses_scipy)),
    losses_scipy, label=f'scipy (final={losses_scipy[-1]:.4g})'
  )
  axes[0].set_xlabel('iteration')
  axes[0].set_ylabel('MSE loss')
  axes[0].set_title('Learning curves')
  axes[0].legend()

  # per-condition fits
  for i, label in enumerate(labels):
    ts = np.array(measurements[label]['timestamps'])
    ys = np.array(measurements[label]['measurements'])
    ax = axes[i + 1]
    ax.scatter(ts, ys, color='black', zorder=3, label='data')
    ax.plot(ts, np.array(preds_euler[label]), color='tab:blue',  marker='o', label='euler')
    ax.plot(ts, np.array(preds_scipy[label]), color='tab:orange', marker='s', label='scipy')
    ax.set_title(label)
    ax.set_xlabel('time')
    ax.legend()

  fig.tight_layout()
  fig.savefig(os.path.join(plot_root, 'euler_vs_scipy.png'))
  plt.close(fig)

  # --- assertions ---
  assert losses_euler[-1] < 0.1, f'euler loss too high: {losses_euler[-1]}'
  assert losses_scipy[-1] < 0.1, f'scipy loss too high: {losses_scipy[-1]}'

  ratio = losses_scipy[-1] / (losses_euler[-1] + 1e-10)
  assert ratio < 10.0, \
    f'scipy loss ({losses_scipy[-1]:.4g}) much worse than euler ({losses_euler[-1]:.4g})'


def test_auto_mode_uses_euler(simple_setup):
  """auto mode should succeed via euler for a well-behaved model."""
  model, conditions, measurements = simple_setup
  parameter_ranges = model.parameter_ranges()

  losses, parameters, predictions = maximum_likelihood_estimate(
    model, conditions, measurements, parameter_ranges, iterations=256, mode='auto'
  )

  assert len(losses) > 0
  assert np.all(np.isfinite(losses))
  assert losses[-1] < 0.1


def test_auto_mode_falls_back_to_scipy(simple_setup):
  """auto mode should fall back to scipy when euler raises an AssertionError."""
  model, conditions, measurements = simple_setup
  parameter_ranges = model.parameter_ranges()

  # force euler to fail by setting error_tol=0 via monkey-patching the assertion
  from unittest.mock import patch

  original = maximum_likelihood_estimate_euler

  def failing_euler(*args, **kwargs):
    raise AssertionError('forced integration error')

  with patch('doe.inference.gradient.maximum_likelihood_estimate_euler', failing_euler):
    losses, parameters, predictions = maximum_likelihood_estimate(
      model, conditions, measurements, parameter_ranges, iterations=50, mode='auto'
    )

  assert len(losses) > 0
  assert np.all(np.isfinite(losses))

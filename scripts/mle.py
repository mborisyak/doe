"""
CLI script for Maximum Likelihood Estimation using a CustomODESystem.

Usage:
  python scripts/mle.py --model MODEL_SPEC.json --conditions CONDITIONS.json --data MEASUREMENTS.json [options]

Example:
  python scripts/mle.py --model data/models/simple.json --conditions data/experiments/example.json --data data/experiments/measurements.json

Arguments:
  --model        Path to JSON file defining the CustomODESystem spec.
  --conditions   Path to JSON file with experimental conditions
                 (dict: label -> {A, B, E, temperature}).
  --data         Path to JSON file with measurements
                 (dict: label -> {timestamps: [...], measurements: [...]}).
  --initial      Optional JSON file with initial parameter values.
                 Accepted forms:
                   {"q": ..., "K_A": ..., "K_B": ...}
                 or
                   {"parameters": {"q": ..., "K_A": ..., "K_B": ...}}
  --dtype        Numeric dtype: float32 or float64 (default: float32).
  --iterations   Max optimiser iterations (default: 1024).
  --rtol         Relative tolerance for early stopping (default: 1e-6).
                 Pass 0 to disable early stopping.
  --output       Write results as JSON to this file instead of stdout.
  --loo          Also run leave-one-out cross-validation over experiments.
  --plot FILE    Save a summary plot to FILE (e.g. plot.png).
"""

import argparse
import json
import sys

import jax
import jax.numpy as jnp

from doe.common import Conditions
from doe.common.custom import CustomODESystem
from doe.inference import maximum_likelihood_estimate


def parse_args():
  p = argparse.ArgumentParser(description='Maximum Likelihood Estimation with CustomODESystem.')
  p.add_argument('--model', required=True, metavar='FILE',
                 help='JSON file with the CustomODESystem spec.')
  p.add_argument('--conditions', required=True, metavar='FILE',
                 help='JSON file with experimental conditions.')
  p.add_argument('--data', required=True, metavar='FILE',
                 help='JSON file with timestamps and measurements per experiment.')
  p.add_argument('--initial', metavar='FILE', default=None,
                 help='Optional JSON file with initial parameter values.')
  p.add_argument('--dtype', choices=('float32', 'float64'), default='float32',
                 help='Numeric dtype for optimisation (default: float32).')
  p.add_argument('--iterations', type=int, default=1024, metavar='N',
                 help='Maximum number of optimiser iterations (default: 1024).')
  p.add_argument('--rtol', type=float, default=1e-6, metavar='TOL',
                 help='Relative tolerance for early stopping (default: 1e-6). Pass 0 to disable.')
  p.add_argument('--output', metavar='FILE', default=None,
                 help='Write JSON results to FILE instead of stdout.')
  p.add_argument('--loo', action='store_true',
                 help='Run leave-one-out cross-validation over experiments.')
  p.add_argument('--plot', metavar='FILE', default=None,
                 help='Save a summary plot to FILE.')
  return p.parse_args()


def main():
  args = parse_args()

  with open(args.model) as f:
    spec = json.load(f)

  with open(args.conditions) as f:
    conditions = json.load(f)

  with open(args.data) as f:
    data = json.load(f)

  # Validate that condition and data labels match
  missing_data = set(conditions) - set(data)
  missing_cond = set(data) - set(conditions)
  if missing_data:
    print(f'error: experiments in conditions but missing from data: {sorted(missing_data)}', file=sys.stderr)
    sys.exit(1)
  if missing_cond:
    print(f'error: experiments in data but missing from conditions: {sorted(missing_cond)}', file=sys.stderr)
    sys.exit(1)

  model = CustomODESystem(spec)
  parameter_ranges = model.parameter_ranges()
  dtype = jnp.float64 if args.dtype == 'float64' else jnp.float32

  rtol = args.rtol if args.rtol != 0.0 else None

  initial = None
  if args.initial is not None:
    with open(args.initial) as f:
      initial_payload = json.load(f)

    if isinstance(initial_payload, dict) and 'parameters' in initial_payload:
      initial_payload = initial_payload['parameters']

    if not isinstance(initial_payload, dict):
      print('error: --initial must be a JSON object or contain top-level "parameters" object', file=sys.stderr)
      sys.exit(1)

    expected = set(model.parameters.keys())
    provided = set(initial_payload.keys())
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    if missing:
      print(f'error: initial parameter file is missing required keys: {missing}', file=sys.stderr)
      sys.exit(1)
    if extra:
      print(f'error: initial parameter file has unexpected keys: {extra}', file=sys.stderr)
      sys.exit(1)

    initial = model.Parameters(**{
      name: float(initial_payload[name])
      for name in model.parameters
    })

  losses, parameters, predictions = maximum_likelihood_estimate(
    model, conditions, data,
    parameter_ranges=parameter_ranges,
    initial=initial,
    iterations=args.iterations,
    rtol=rtol,
    dtype=dtype,
  )

  residuals = {}
  for label, pred in predictions.items():
    measured = data[label]['measurements']
    residuals[label] = [float(p - m) for p, m in zip(pred, measured)]

  result = {
    'parameters': {name: float(getattr(parameters, name)) for name in model.parameters},
    'loss_trace': [float(v) for v in losses],
    'loss': float(losses[-1]),
    'iterations': int(len(losses)),
    'predictions': {label: [float(v) for v in pred] for label, pred in predictions.items()},
    'residuals': residuals,
    'rmse': {label: (sum(r ** 2 for r in res) / len(res)) ** 0.5 for label, res in residuals.items()},
  }

  if args.loo:
    labels = list(conditions.keys())
    loo_predictions = {}
    for held_out in labels:
      print(f'LOO: holding out {held_out!r} ...', file=sys.stderr)
      train_conditions = {k: v for k, v in conditions.items() if k != held_out}
      train_data = {k: v for k, v in data.items() if k != held_out}

      _, params_loo, _ = maximum_likelihood_estimate(
        model, train_conditions, train_data,
        parameter_ranges=parameter_ranges,
        iterations=args.iterations,
        rtol=rtol,
      )

      held_cond = Conditions(**{
        k: jnp.array([conditions[held_out][k]], dtype=jnp.float32)
        for k in Conditions._fields
      })
      held_ts = jnp.array([data[held_out]['timestamps']], dtype=jnp.float32)
      pred = jax.vmap(model.solve, in_axes=(0, 0, None))(held_cond, held_ts, params_loo)[0]
      loo_predictions[held_out] = [float(v) for v in pred]

    loo_residuals = {
      label: [float(p - m) for p, m in zip(loo_predictions[label], data[label]['measurements'])]
      for label in labels
    }
    result['loo_predictions'] = loo_predictions
    result['loo_residuals'] = loo_residuals
    result['loo_rmse'] = {
      label: (sum(r ** 2 for r in res) / len(res)) ** 0.5
      for label, res in loo_residuals.items()
    }

  if args.plot:
    import matplotlib.pyplot as plt

    exp_labels = list(conditions.keys())
    n = len(exp_labels)
    ncols = max(2, n)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 10))

    # --- loss trace ---
    ax = axes[0, 0]
    mean_rmse = sum(result['rmse'].values()) / len(result['rmse'])
    ax.plot(losses)
    ax.set_title(f'Optimisation loss\nmean RMSE={mean_rmse:.4f}')
    ax.set_xlabel('iteration')
    ax.set_ylabel('MSE')

    # --- parameter bar chart (normalized to [0, 1] within ranges) ---
    ax = axes[0, 1]
    param_names = list(model.parameters.keys())
    norm_values = [
      (float(getattr(parameters, name)) - low) / (high - low)
      for name, (low, high) in model.parameters.items()
    ]
    ax.bar(param_names, norm_values)
    ax.set_ylim(0, 1)
    ax.set_ylabel('normalized value')
    ax.set_title('Fitted parameters')
    ax.axhline(0.5, color='grey', linewidth=0.8, linestyle='--')

    for j in range(2, ncols):
      axes[0, j].set_visible(False)

    # --- data + fit per experiment ---
    for i, label in enumerate(exp_labels):
      ax = axes[1, i]
      ts = data[label]['timestamps']
      meas = data[label]['measurements']
      pred = result['predictions'][label]
      rmse = result['rmse'][label]

      ax.scatter(ts, meas, label='data', zorder=5, s=30)
      ax.plot(ts, pred, label=f'fit  RMSE={rmse:.4f}')

      if 'loo_predictions' in result:
        loo_pred = result['loo_predictions'][label]
        loo_rmse = result['loo_rmse'][label]
        ax.plot(ts, loo_pred, '--', label=f'LOO  RMSE={loo_rmse:.4f}')

      ax.set_title(label)
      ax.set_xlabel('time')
      ax.legend(fontsize=8)

    for j in range(n, ncols):
      axes[1, j].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.plot)
    plt.close(fig)
    print(f'Plot saved to {args.plot}', file=sys.stderr)

  output = json.dumps(result, indent=2)
  if args.output is None:
    print(output)
  else:
    with open(args.output, 'w') as f:
      f.write(output)
    print(f'Results written to {args.output}', file=sys.stderr)


if __name__ == '__main__':
  main()

### NOTE: a draft of OED CLI, use new_exp.py instead.

"""
CLI script for Design of Experiments using the Fisher information criterion.

Usage:
  python scripts/doe.py --model MODEL_SPEC.json --condition-ranges RANGES.json
    (--parameters PARAMS.json | --mle) --timestamps T [T ...] [options]

Example:
  python scripts/doe.py --model data/models/simple.json \
    --condition-ranges data/experiments/condition_ranges.json \
    --parameters data/experiments/parameters.json \
    --timestamps 3 6 9 12 15 18 21 24 27

Arguments:
  --model              JSON file defining the CustomODESystem spec.
  --condition-ranges   JSON file with condition ranges ({field: [low, high]}).
  --parameters         JSON file with parameter values ({name: value}).
  --mle                Infer parameters from history using MLE (requires --history-*).
  --history-conditions JSON file with past experimental conditions.
  --history-data       JSON file with past measurements.
  --timestamps         Measurement timestamps for proposed experiments (space-separated).
  --n                  Number of experiments to propose (default: 1).
  --iterations         Fisher optimiser iterations (default: 64).
  --mle-iterations     MLE optimiser iterations when --mle is used (default: 1024).
  --criterion          Optimality criterion: D or A (default: D).
  --seed               Random seed (default: 0).
  --output             Write JSON results to this file instead of stdout.
  --plot               Save a summary plot to FILE.
"""

import argparse
import json
import sys

import jax
import jax.numpy as jnp

from doe.common import Conditions
from doe.common.custom import CustomODESystem
from doe.doe import Fisher
from doe.inference import maximum_likelihood_estimate


def parse_args():
  p = argparse.ArgumentParser(description='Design of Experiments with Fisher information.')
  p.add_argument('--model', required=True, metavar='FILE',
                 help='JSON file with the CustomODESystem spec.')
  p.add_argument('--condition-ranges', required=True, metavar='FILE',
                 help='JSON file with condition ranges ({field: [low, high]}).')

  param_group = p.add_mutually_exclusive_group(required=True)
  param_group.add_argument('--parameters', metavar='FILE',
                           help='JSON file with parameter values ({name: value}).')
  param_group.add_argument('--mle', action='store_true',
                           help='Infer parameters from history using MLE.')

  p.add_argument('--history-conditions', metavar='FILE', default=None,
                 help='JSON file with past experimental conditions.')
  p.add_argument('--history-data', metavar='FILE', default=None,
                 help='JSON file with past measurements.')

  p.add_argument('--timestamps', type=float, nargs='+', required=True, metavar='T',
                 help='Measurement timestamps for proposed experiments.')
  p.add_argument('--n', type=int, default=1, metavar='N',
                 help='Number of experiments to propose (default: 1).')
  p.add_argument('--iterations', type=int, default=64, metavar='N',
                 help='Fisher optimiser iterations (default: 64).')
  p.add_argument('--mle-iterations', type=int, default=1024, metavar='N',
                 help='MLE optimiser iterations when --mle is used (default: 1024).')
  p.add_argument('--criterion', choices=['D', 'A'], default='D',
                 help='Optimality criterion: D or A (default: D).')
  p.add_argument('--seed', type=int, default=0,
                 help='Random seed (default: 0).')
  p.add_argument('--output', metavar='FILE', default=None,
                 help='Write JSON results to FILE instead of stdout.')
  p.add_argument('--plot', metavar='FILE', default=None,
                 help='Save a summary plot to FILE.')
  return p.parse_args()


def main():
  args = parse_args()

  with open(args.model) as f:
    spec = json.load(f)

  with open(args.condition_ranges) as f:
    condition_ranges_data = json.load(f)

  has_history = args.history_conditions is not None or args.history_data is not None

  if args.mle and not has_history:
    print('error: --mle requires --history-conditions and --history-data', file=sys.stderr)
    sys.exit(1)

  if has_history and (args.history_conditions is None or args.history_data is None):
    print('error: --history-conditions and --history-data must be provided together', file=sys.stderr)
    sys.exit(1)

  if has_history:
    with open(args.history_conditions) as f:
      history_conditions = json.load(f)
    with open(args.history_data) as f:
      history_data = json.load(f)
  else:
    history_conditions = {}
    history_data = {}

  model = CustomODESystem(spec)
  parameter_ranges = model.parameter_ranges()

  condition_ranges = Conditions(**{k: condition_ranges_data[k] for k in Conditions._fields})
  timestamps = jnp.array(args.timestamps, dtype=jnp.float32)
  dtype = jnp.float32

  if args.mle:
    print('Running MLE...', file=sys.stderr)
    _, parameters, _ = maximum_likelihood_estimate(
      model, history_conditions, history_data,
      parameter_ranges=parameter_ranges,
      iterations=args.mle_iterations,
    )
  else:
    with open(args.parameters) as f:
      params_data = json.load(f)
    parameters = model.Parameters(**{name: params_data[name] for name in model.parameters})

  fisher = Fisher(
    model, condition_ranges, parameter_ranges,
    timestamps=timestamps,
    iterations=args.iterations,
    criterion=args.criterion,
  )

  label_order = list(history_conditions.keys())
  if label_order:
    controls = jnp.stack([
      fisher.encode_conditions(Conditions(**{
        k: jnp.array(history_conditions[label][k], dtype=dtype)
        for k in Conditions._fields
      }))
      for label in label_order
    ])
    timestamps_hist = jnp.array(
      [history_data[label]['timestamps'] for label in label_order], dtype=dtype
    )
  else:
    controls = jnp.zeros((0, len(Conditions._fields)), dtype=dtype)
    timestamps_hist = jnp.zeros((0, len(args.timestamps)), dtype=dtype)

  key = jax.random.PRNGKey(args.seed)
  loss_trace, proposed = fisher.propose(
    key, n=args.n, controls=controls, timestamps=timestamps_hist, parameters=parameters
  )

  proposed_conditions = {
    f'experiment {i + 1}': {
      k: float(getattr(fisher.decode_conditions(proposed[i]), k))
      for k in Conditions._fields
    }
    for i in range(args.n)
  }

  result = {
    'proposed': proposed_conditions,
    'parameters': {name: float(getattr(parameters, name)) for name in model.parameters},
    'loss': float(loss_trace[-1]),
    'iterations': int(len(loss_trace)),
  }

  if args.plot:
    import matplotlib.pyplot as plt

    n = args.n
    ncols = max(2, n)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8))

    # --- Fisher loss trace ---
    ax = axes[0, 0]
    ax.plot(loss_trace)
    ax.set_title(f'Fisher loss ({args.criterion}-optimality)\nfinal: {loss_trace[-1]:.4f}')
    ax.set_xlabel('iteration')
    ax.set_ylabel(f'{args.criterion}-criterion')

    # --- parameters bar chart (normalized) ---
    ax = axes[0, 1]
    param_names = list(model.parameters.keys())
    norm_params = [
      (float(getattr(parameters, name)) - low) / (high - low)
      for name, (low, high) in model.parameters.items()
    ]
    ax.bar(param_names, norm_params)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='grey', linewidth=0.8, linestyle='--')
    ax.set_ylabel('normalized value')
    ax.set_title('Parameters used')

    for j in range(2, ncols):
      axes[0, j].set_visible(False)

    # --- proposed conditions per experiment ---
    cond_fields = list(Conditions._fields)
    for i in range(n):
      ax = axes[1, i]
      label = f'experiment {i + 1}'
      cond = proposed_conditions[label]
      norm_cond = [
        (cond[k] - condition_ranges_data[k][0]) / (condition_ranges_data[k][1] - condition_ranges_data[k][0])
        for k in cond_fields
      ]
      ax.bar(cond_fields, norm_cond)
      ax.set_ylim(0, 1)
      ax.axhline(0.5, color='grey', linewidth=0.8, linestyle='--')
      ax.set_title(f'Proposed: {label}')
      ax.set_ylabel('normalized value')

      # annotate with actual values
      for j, (k, v) in enumerate(cond.items()):
        ax.text(j, norm_cond[j] + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

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
"""
CLI script for proposing new informative experiments using Fisher D-optimality.

Usage:
  python scripts/new_exp.py --model MODEL_SPEC.json --conditions CONDITIONS.json --data MEASUREMENTS.json --parameters PARAMETERS.json --condition-ranges CONDITION_RANGES.json [options]

Example:
  python scripts/new_exp.py \
    --model data/secret/simple.json \
    --conditions data/experiments/example.json \
    --data data/experiments/measurements.json \
    --parameters fitted.json \
    --condition-ranges ranges.json \
    --n 3 --iterations 64 --seed 0

Arguments:
  --model             Path to JSON file defining the CustomODESystem spec.
  --conditions        Path to JSON file with historical experimental conditions
                      (dict: label -> {A, B, E, temperature}).
  --data              Path to JSON file with historical measurements
                      (dict: label -> {timestamps: [...], measurements: [...]}).
                      Only timestamps are used by this script.
  --parameters        Path to JSON with fitted parameter values.
                      Accepted forms:
                        {"q": ..., "K_A": ..., "K_B": ...}
                      or
                        {"parameters": {"q": ..., "K_A": ..., "K_B": ...}}
  --condition-ranges  Path to JSON with bounds for proposed conditions:
                      {"A": [low, high], "B": [low, high], "E": [low, high], "temperature": [low, high]}
  --n                 Number of new experiments to propose (default: 3).
  --iterations        Number of Fisher optimizer steps (default: 64).
  --criterion         Fisher criterion: D or A (default: D).
  --regularization    Optional L2 regularization strength for encoded controls.
  --seed              PRNG seed (default: 0).
  --proposal-timestamps
                      Timestamps to optimize for new experiments. If omitted,
                      uses timestamps from the first historical experiment.
  --output            Write results as JSON to this file instead of stdout.
  --plot FILE         Save a summary plot to FILE (e.g. fisher.png).
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

from doe.common import Conditions
from doe.common.custom import CustomODESystem
from doe.doe import Fisher
from doe.inference import maximum_likelihood_estimate


def parse_args():
  p = argparse.ArgumentParser(description='Propose informative experiments with Fisher optimal design.')
  p.add_argument('--model', required=True, metavar='FILE',
                 help='JSON file with the CustomODESystem spec.')
  p.add_argument('--conditions', required=False, default=None, metavar='FILE',
                 help='JSON file with historical experimental conditions.')
  p.add_argument('--data', required=False, default=None, metavar='FILE',
                 help='JSON file with historical measurements (timestamps are used).')
  p.add_argument('--parameters', required=False, default=None, metavar='FILE',
                 help='JSON file with fitted parameter values.')
  p.add_argument('--n', type=int, default=None, metavar='N',
                 help='Number of new experiments to propose (default: config value).')
  p.add_argument('--iterations', type=int, default=64, metavar='N',
                 help='Number of Fisher optimisation steps (default: 64).')
  p.add_argument('--criterion', choices=('D', 'A'), default='D',
                 help='Fisher design criterion: D or A (default: D).')
  p.add_argument('--regularization', type=float, default=None, metavar='LAMBDA',
                 help='Optional L2 regularization strength.')
  p.add_argument('--seed', type=int, default=0, metavar='SEED',
                 help='PRNG seed (default: 0).')
  p.add_argument('--config', metavar='FILE', default=None,
                 help='Path to config.yaml (default: <root>/config/config.yaml).')
  p.add_argument('--output', metavar='FILE', default=None,
                 help='Write JSON results to FILE instead of stdout.')
  p.add_argument('--plot', metavar='FILE', default=None,
                 help='Save a summary plot to FILE.')
  p.add_argument('--device', metavar='FILE', default='cpu',
                 help='Compilation target.')
  return p.parse_args()


def _fail(message):
  print(f'error: {message}', file=sys.stderr)
  sys.exit(1)


def _load_json(path, description):
  try:
    with open(path) as f:
      return json.load(f)
  except FileNotFoundError:
    _fail(f'{description} file does not exist: {path}')
  except json.JSONDecodeError as exc:
    _fail(f'failed to parse {description} as JSON ({path}): {exc}')


def _validate_finite(value, field_path):
  try:
    x = float(value)
  except (TypeError, ValueError):
    _fail(f'{field_path} must be numeric; got {value!r}')
  if not math.isfinite(x):
    _fail(f'{field_path} must be finite; got {value!r}')
  return x


def _load_parameters(path, model):
  payload = _load_json(path, 'parameters')

  if isinstance(payload, dict) and 'parameters' in payload:
    payload = payload['parameters']

  if not isinstance(payload, dict):
    _fail('parameters JSON must be an object or contain a top-level "parameters" object')

  expected = list(model.parameters.keys())
  missing = sorted(set(expected) - set(payload))
  extra = sorted(set(payload) - set(expected))

  if missing:
    _fail(f'parameters file is missing required parameters: {missing}')
  if extra:
    _fail(f'parameters file contains unexpected parameters: {extra}')

  values = {
    name: _validate_finite(payload[name], f'parameters.{name}')
    for name in expected
  }
  return model.Parameters(**values)


def _extract_history(conditions, data):
  if not isinstance(conditions, dict) or not conditions:
    _fail('conditions JSON must be a non-empty object: label -> condition')
  if not isinstance(data, dict) or not data:
    _fail('data JSON must be a non-empty object: label -> experiment')

  missing_data = sorted(set(conditions) - set(data))
  missing_cond = sorted(set(data) - set(conditions))
  if missing_data:
    _fail(f'experiments in conditions but missing from data: {missing_data}')
  if missing_cond:
    _fail(f'experiments in data but missing from conditions: {missing_cond}')

  labels = list(conditions.keys())

  history_conditions = []
  history_timestamps = []
  n_timestamps = None

  for label in labels:
    cond = conditions[label]
    if not isinstance(cond, dict):
      _fail(f'conditions.{label} must be an object')

    missing_fields = sorted(set(Conditions._fields) - set(cond))
    extra_fields = sorted(set(cond) - set(Conditions._fields))
    if missing_fields:
      _fail(f'conditions.{label} is missing fields: {missing_fields}')
    if extra_fields:
      _fail(f'conditions.{label} contains unexpected fields: {extra_fields}')

    parsed_cond = Conditions(**{
      name: _validate_finite(cond[name], f'conditions.{label}.{name}')
      for name in Conditions._fields
    })
    history_conditions.append(parsed_cond)

    if 'timestamps' not in data[label]:
      _fail(f'data.{label} must contain "timestamps"')

    ts = data[label]['timestamps']
    ys = data[label]['measurements']
    if not isinstance(ts, list) or not ts:
      _fail(f'data.{label}.timestamps must be a non-empty list')

    parsed_ts = [
      _validate_finite(t, f'data.{label}.timestamps[{i}]')
      for i, t in enumerate(ts)
    ]

    parsed_ys = [
      _validate_finite(y, f'data.{label}.measurements[{i}]')
      for i, y in enumerate(ys)
    ]

    if n_timestamps is None:
      n_timestamps = len(parsed_ts)
    elif len(parsed_ts) != n_timestamps:
      _fail('all historical experiments must use the same number of timestamps')

    if len(parsed_ys) != n_timestamps:
      _fail('all historical experiments must use the same number of timestamps')

    history_timestamps.append(parsed_ts)

  return labels, history_conditions, history_timestamps


def _validate_parameters_in_ranges(parameters, parameter_ranges):
  for name in parameter_ranges._fields:
    value = float(getattr(parameters, name))
    low, high = [float(v) for v in getattr(parameter_ranges, name)]
    if value < low or value > high:
      _fail(f'parameter {name}={value} is outside model range [{low}, {high}]')


def _validate_history_conditions_in_ranges(history_conditions, condition_ranges):
  for idx, cond in enumerate(history_conditions):
    for name in Conditions._fields:
      value = float(getattr(cond, name))
      low, high = [float(v) for v in getattr(condition_ranges, name)]
      if value < low or value > high:
        _fail(
          f'historical condition {idx + 1} field {name}={value} is outside '
          f'condition range [{low}, {high}]'
        )


def _plot_summary(path, loss_trace, proposals, ranges, expected, timestamps):
  import matplotlib.pyplot as plt

  n = len(proposals)
  fields = list(Conditions._fields)

  fig, axes = plt.subplots(3, n, figsize=(5 * n, 8), squeeze=False)

  axes[0, 0].plot(loss_trace)
  axes[0, 0].set_title('Fisher loss')
  axes[0, 0].set_xlabel('iteration')

  for j in range(1, n):
    axes[0, j].set_visible(False)

  for i, proposal in enumerate(proposals):
    normalized = []
    for field in fields:
      low, high = getattr(ranges, field)
      normalized.append(
        (proposal[field] - low) / (high - low)
      )
    axes[1, i].bar(fields, normalized)
    axes[1, i].set_title(f'Proposed conditions {i + 1}')
    axes[1, i].set_ylim([0.0, 1.05])

  y_min, y_max = np.min(expected), np.max(expected)
  delta = y_max - y_min
  y_min, y_max = y_min - 0.025 * delta, y_max + 0.025 * delta

  for i, trajectory in enumerate(expected):
    axes[2, i].plot(timestamps, trajectory)
    axes[2, i].set_ylim([y_min, y_max])

  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)


def main():
  args = parse_args()

  if args.config is None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root, 'config', 'config.yaml')
  else:
    config_path = args.config
  with open(config_path) as f:
    import yaml
    config = yaml.safe_load(f)

  if args.n is None:
    n_batch = config['experiment']['batch']
  else:
    n_batch = args.n

  if n_batch < 1:
    _fail('--n must be >= 1')
  if args.iterations < 1:
    _fail('--iterations must be >= 1')
  if args.regularization is not None and args.regularization < 0:
    _fail('--regularization must be >= 0 when provided')

  device, *_ = jax.devices(args.device)

  spec = _load_json(args.model, 'model')

  if args.conditions is None:
    assert args.data is None, 'conditions and data must be None simultaneously'
    conditions, data = {}, {}
    labels, history_conditions, history_timestamps = [], [], []
  else:
    assert args.data is not None, 'data file must provided with condition file'

    conditions = _load_json(args.conditions, 'conditions')
    data = _load_json(args.data, 'data')
    labels, history_conditions, history_timestamps = _extract_history(conditions, data)

  model = CustomODESystem(spec, device=device)
  parameter_ranges = model.parameter_ranges()
  if args.parameters is None:
    if len(data) > 0:
      print('Parameters are not provided, fitting')
      _, parameters, _ = maximum_likelihood_estimate(
        model, conditions, data, model.parameter_ranges(),
        mode='auto', iterations=None, rtol=1.0e-3
      )
    else:
      print('Parameters are not provided, data is not provided, assuming center of the ranges')
      parameters = model.Parameters(*(
        (low + high) / 2 for low, high in model.parameter_ranges()
      ))
  else:
    parameters = _load_parameters(args.parameters, model)
  _validate_parameters_in_ranges(parameters, parameter_ranges)

  condition_ranges = Conditions(**{
    name: config['conditions'][name]
    for name in Conditions._fields
  })

  T = config['experiment']['duration']
  n = config['experiment']['measurements']
  proposal_timestamps = np.linspace(0, T, num=n + 2)[1:-1]

  dtype = jnp.float32
  fisher = Fisher(
    model,
    condition_ranges=condition_ranges,
    parameter_ranges=parameter_ranges,
    timestamps=jnp.array(proposal_timestamps, dtype=dtype),
    iterations=args.iterations,
    regularization=args.regularization,
    criterion=args.criterion,
  )

  key = jax.random.PRNGKey(args.seed)

  loss_trace, proposal = fisher.propose(
    key=key,
    n=n_batch,
    controls=history_conditions,
    timestamps=history_timestamps,
    parameters=parameters,
  )

  expected = [
    model.solve(jax.tree.map(lambda x: x[i], proposal), proposal_timestamps, parameters)
    for i in range(n_batch)
  ]
  expected = [[float(y) for y in ys] for ys in expected]

  proposals = []
  for i in range(n_batch):
    cs = jax.tree.map(lambda x: x[i], proposal)
    as_dict = {
      name: float(getattr(cs, name))
      for name in Conditions._fields
    }
    proposals.append(as_dict)

  result = {
    'criterion': args.criterion,
    'seed': args.seed,
    'iterations': args.iterations,
    'n_proposals': n_batch,
    'proposal_timestamps': [float(t) for t in proposal_timestamps],
    'parameters': {
      name: float(getattr(parameters, name))
      for name in model.parameters
    },
    'condition_ranges': {
      name: [float(v) for v in getattr(condition_ranges, name)]
      for name in Conditions._fields
    },
    'proposals': proposals,
    'expected': expected
  }

  if args.plot:
    _plot_summary(args.plot, loss_trace, proposals, condition_ranges, expected, proposal_timestamps)
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

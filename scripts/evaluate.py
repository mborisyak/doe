### evaluation script
### usage: python scripts/evaluate.py --model data/models/super-model-1.json --parameters data/models/super-model-1.parameters.json --conditions data/test_conditions.json --ground-truth data/test.json
### fixed test dataset is in the repository.
### to sample parameters from provided ranges, simply omit the --parameters flag.

import argparse
import json
import math
import sys

import numpy as np

import jax
from jax import config; config.update("jax_enable_x64", False); config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

from tqdm import tqdm

from doe.common import Conditions
from doe.common.custom import CustomODESystem

NOISE_STD = 0.025

def get_hashed_seed(name):
  import hashlib

  h = hashlib.sha256()
  h.update(bytes(name, encoding='utf-8'))
  digest = h.hexdigest()

  return int(digest[:8], 16)

SUPER_SECRET_SEED = get_hashed_seed('SUPER_SECRET_SEED')
N_TEST_PARAMETERS = 128

def parse_args():
  p = argparse.ArgumentParser(description='Maximum Likelihood Estimation with CustomODESystem.')
  p.add_argument('--model', required=True, metavar='FILE',
                 help='JSON file with the CustomODESystem spec.')
  p.add_argument('--parameters', required=False, default=None, metavar='FILE',
                 help='JSON file with values of model parameters.')
  p.add_argument('--conditions', required=True, metavar='FILE',
                 help='JSON file with experimental conditions.')
  p.add_argument('--ground-truth', required=True, metavar='FILE',
                 help='JSON file with ground-truth measurements.')
  p.add_argument('--dtype', choices=('float32', 'float64'), default='float32',
                 help='Numeric dtype for optimisation (default: float32).')
  p.add_argument('--output', metavar='FILE', default=None,
                 help='Write JSON results to FILE instead of stdout.')
  p.add_argument('--cores', metavar='FILE', default=None, type=int,
                 help='Number of cores for evaluation, used if parameters are not provided.')
  p.add_argument('--plot', metavar='FILE', default=None, type=int,
                 help='plots 9 worst test cases.')
  return p.parse_args()

def solve(arguments, progress=False):
  spec, params, conditions, timestamps, dtype = arguments

  model = CustomODESystem(spec)
  params = model.Parameters(**{k: jnp.array(v, dtype=dtype) for k, v in params.items()})

  _solve = jax.jit(model.solve_euler)
  # cpu, *_ = jax.devices('cpu')
  # _solve = lambda args, **kwargs: model.solve_scipy(args, **kwargs, device=cpu)

  result, errors = [], []

  progress_bar = tqdm if progress else (lambda x: x)

  for label, condition in progress_bar(conditions.items()):
    condition = Conditions(**{k: jnp.array(v, dtype=dtype) for k, v in conditions[label].items()})
    trajectory, error = _solve(condition, timestamps=timestamps[label], parameters=params)
    result.append(trajectory)
    errors.append(error)

  result = np.stack(result, axis=0)
  errors = np.stack(errors, axis=0)
  return result, errors

def main():
  args = parse_args()
  dtype = jnp.float64 if args.dtype == 'float64' else jnp.float32

  with open(args.model, 'r') as f:
    spec = json.load(f)

  parameter_ranges = spec['parameters']
  if args.parameters is None:
    parameters = []
    rng = np.random.default_rng(SUPER_SECRET_SEED)
    for i in range(N_TEST_PARAMETERS):
      params = {
        k: rng.uniform(low=low, high=high, size=())
        for k, (low, high) in parameter_ranges.items()
      }
      parameters.append(params)

  else:
    with open(args.parameters, 'r') as f:
      estimation = json.load(f)['parameters']
      parameters = [estimation]

  with open(args.conditions, 'r') as f:
    conditions = json.load(f)

  with open(args.ground_truth, 'r') as f:
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

  import multiprocessing as mp
  timestamps = {
    label: np.array(data[label]['timestamps'], dtype=dtype)
    for label in data
  }

  test_label, *_ = conditions
  model = CustomODESystem(spec)
  test_conditions = Conditions(**{k: jnp.array(v, dtype=dtype) for k, v in conditions[test_label].items()})
  params = model.Parameters(**parameters[0])
  rhs = model.rhs(model.initial_state(test_conditions), test_conditions, params)

  if len(parameters) > 1:
    context = mp.get_context('spawn')
    pool = context.Pool(processes=args.cores, )
    evaluated = pool.map(
      solve, (
        (spec, params, conditions, timestamps, dtype)
        for params in parameters
      )
    )
    predictions, errors = zip(*evaluated)
  else:
    predictions, errors = zip(*[
      solve((spec, params, conditions, timestamps, dtype), progress=True)
      for params in parameters
    ])

  if np.max(errors) > 1.0e-3:
    print(f'WARNING: integration error {np.max(errors):.3}')

  predictions = np.stack(predictions, axis=0)
  measurements = np.stack([
    data[label]['measurements']
    for label in conditions
  ], axis=0)

  mse = np.mean(np.square(predictions - measurements[None]))
  rmse = np.sqrt(mse)

  variance = np.var([y for label in data for y in data[label]['measurements']])
  r_score = 1 - mse / variance

  print(f'[{args.model}]')
  print(f'test RMSE: {rmse:.2f}')
  print(f'std labels: {np.sqrt(variance):.2f}')
  print(f'R^2 = {r_score:.2f}')

  C = np.stack([
    [condition[k] for k in Conditions._fields]
    for _, condition in conditions.items()
  ], axis=0)

  mse_per_experiment = np.mean(np.square(predictions - measurements[None]), axis=(0, 2))

  from sklearn.ensemble import GradientBoostingRegressor

  reg = GradientBoostingRegressor(n_estimators=20, max_depth=5)
  reg.fit(C, mse_per_experiment)

  print('Error prediction importance (GBDT):')
  importance = reg.feature_importances_
  for i, k in enumerate(Conditions._fields):
    print(f'  {k}: {importance[i]:.2f}')

  worst_index = np.argsort(mse_per_experiment)[-9:]
  labels = [label for label in conditions]

  import matplotlib.pyplot as plt
  fig = plt.figure(figsize=(15, 15))
  axes = fig.subplots(3, 3).ravel()

  for i in range(9):
    k = worst_index[i]
    label = labels[k]
    T = np.max(timestamps[label])
    ts = np.linspace(0, T, num=1024)
    pred, _ = zip(*[
      solve((spec, params, {label: conditions[label]}, {label: ts}, dtype), progress=False)
      for params in parameters
    ])
    pred = np.stack(pred, axis=0)

    axes[i].scatter(timestamps[label], measurements[k])
    axes[i].plot(ts, pred[:, 0].T)
    conditions_string = ', '.join(f'{f} = {conditions[label][f]:.2f}' for f in Conditions._fields)
    axes[i].set_title(f'{label}\n{conditions_string}')

  fig.tight_layout()
  fig.savefig('test.png')
  plt.close(fig)

if __name__ == '__main__':
  main()

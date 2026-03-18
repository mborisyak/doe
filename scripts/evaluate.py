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
  return p.parse_args()

def solve(arguments):
  spec, params, conditions, timestamps, dtype = arguments

  model = CustomODESystem(spec)
  params = model.Parameters(**{k: jnp.array(v, dtype=dtype) for k, v in params.items()})

  _solve = jax.jit(model.solve)

  result = []

  for label, condition in conditions.items():
    condition = Conditions(**{k: jnp.array(v, dtype=dtype) for k, v in conditions[label].items()})
    trajectory = _solve(condition, timestamps=timestamps[label], parameters=params)
    result.append(trajectory)

  result = np.stack(result, axis=0)
  return result

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
      estimation = json.load(f)
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

  if len(parameters) > 1:
    context = mp.get_context('spawn')
    pool = context.Pool(processes=args.cores, )
    predictions = pool.map(
      solve, (
        (spec, params, conditions, timestamps, dtype)
        for params in parameters
      )
    )
  else:
    predictions = [solve(spec, params, conditions, timestamps, dtype) for params in parameters]

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


if __name__ == '__main__':
  main()

### evaluation script
### usage: python scripts/evaluate.py --model data/models/super-model-1.json --parameters data/models/super-model-1.parameters.json --conditions data/test_conditions.json --ground-truth data/test.json
### fixed test dataset is in the repository.

import argparse
import json
import math
import sys

import numpy as np

import jax
import jax.numpy as jnp

from doe.common import Conditions
from doe.common.custom import CustomODESystem
from tqdm import tqdm

NOISE_STD = 0.025

def parse_args():
  p = argparse.ArgumentParser(description='Maximum Likelihood Estimation with CustomODESystem.')
  p.add_argument('--model', required=True, metavar='FILE',
                 help='JSON file with the CustomODESystem spec.')
  p.add_argument('--parameters', required=True, metavar='FILE',
                 help='JSON file with values of model parameters.')
  p.add_argument('--conditions', required=True, metavar='FILE',
                 help='JSON file with experimental conditions.')
  p.add_argument('--ground-truth', required=True, metavar='FILE',
                 help='JSON file with ground-truth measurements.')
  p.add_argument('--dtype', choices=('float32', 'float64'), default='float32',
                 help='Numeric dtype for optimisation (default: float32).')
  p.add_argument('--output', metavar='FILE', default=None,
                 help='Write JSON results to FILE instead of stdout.')
  return p.parse_args()


def main():
  args = parse_args()

  with open(args.model, 'r') as f:
    spec = json.load(f)

  with open(args.parameters, 'r') as f:
    estimation = json.load(f)
    parameters = estimation['parameters']

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

  dtype = jnp.float64 if args.dtype == 'float64' else jnp.float32

  model = CustomODESystem(spec)
  assert set(model.parameters) == set(parameters)
  parameters = model.Parameters(**{
    k: jnp.array(v, dtype=dtype)
    for k, v in parameters.items()
  })

  solve = jax.jit(model.solve)

  predictions = {}
  for label in tqdm(conditions):
    condition = Conditions(**{ k: jnp.array(v, dtype=dtype) for k, v in conditions[label].items() })
    pred = solve(condition, timestamps=data[label]['timestamps'], parameters=parameters)
    predictions[label] = pred

  mse_per_experiment = {}
  for label in predictions:
    ys = np.array(data[label]['measurements'])
    ps = np.array(predictions[label])

    assert ps.shape == ys.shape == (9, )

    mse_per_experiment[label] = np.sum(np.square(ps - ys))

  total_mse = np.sum([m for _, m in mse_per_experiment.items()])
  total_n = sum(len(ps) for _, ps in predictions.items())
  mse = total_mse / total_n
  rmse = np.sqrt(total_mse / total_n)

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

  mse = np.array([mse_per_experiment[label] for label in conditions])

  from sklearn.ensemble import GradientBoostingRegressor

  reg = GradientBoostingRegressor(n_estimators=10, max_depth=3)
  reg.fit(C, mse)

  print('Error prediction importance (GBDT):')
  importance = reg.feature_importances_
  for i, k in enumerate(Conditions._fields):
    print(f'  {k}: {importance[i]}')


if __name__ == '__main__':
  main()

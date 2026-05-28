"""
Fit a Gaussian-process surrogate (fixed kernel hyper-parameters) to batch data.

Store-free, file in / result out (the MCP layer resolves refs, locks, and commits).
The GP maps (conditions, t) -> the observable's value; see doe.gp.surrogate. Reports a
k-fold cross-validation RMSE, then refits on all data and serialises the GP State.

Usage:
  python scripts/gp_fit.py --model GP_SPEC.json --batches BATCH.json [BATCH.json ...] --output RESULT.json

Arguments:
  --model    JSON file with the GP spec (kernel, length_scale(s), variance, noise, observables).
  --batches  One or more batch JSON files.
  --folds    Cross-validation folds (default 5; clamped to the number of points).
  --output   Write JSON results to FILE instead of stdout.
"""
import argparse
import json
import sys

from doe.dataset import assemble
from doe.gp.surrogate import dataset_xy, fit_surrogate, observable_of


def parse_args():
  p = argparse.ArgumentParser(description='Fit a GP surrogate to batch data.')
  p.add_argument('--model', required=True, metavar='FILE', help='GP spec JSON file.')
  p.add_argument('--batches', required=True, nargs='+', metavar='FILE', help='Batch JSON files.')
  p.add_argument('--folds', type=int, default=5, metavar='K', help='Cross-validation folds (default 5).')
  p.add_argument('--output', metavar='FILE', default=None, help='Write JSON results to FILE.')
  return p.parse_args()


def main():
  args = parse_args()
  with open(args.model) as f:
    spec = json.load(f)

  observable = observable_of(spec)  # asserts a single observable
  X, y = dataset_xy(assemble(args.batches), observable)
  if len(y) == 0:
    print(f'error: no data measures observable {observable!r} in {args.batches}', file=sys.stderr)
    sys.exit(1)

  fit = fit_surrogate(spec, X, y, folds=args.folds)
  # The script owns the fitted_model record body; the MCP layer dumps this and stamps
  # linkage refs (model, fit.data, fit.tool_result). predict_gp reads auxiliary.gp_state.
  result = {
    'parameters': {'cv_rmse': fit['cv_rmse']},
    'fit': {'cv_rmse': fit['cv_rmse'], 'n_train': fit['n_train'], 'observable': fit['observable']},
    'auxiliary': {'gp_state': fit['state'], 'predictions': fit['predictions']},
  }

  output = json.dumps(result, indent=2)
  if args.output is None:
    print(output)
  else:
    with open(args.output, 'w') as f:
      f.write(output)
    print(f'Results written to {args.output}', file=sys.stderr)


if __name__ == '__main__':
  main()

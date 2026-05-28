"""
Optimise GP kernel hyper-parameters (per-dimension length-scales + variance) to batch data
by maximising the log-marginal-likelihood. Store-free; file in / result out.

Usage:
  python scripts/gp_hyperfit.py --model BASE_SPEC.json --batches BATCH.json [...] --output RESULT.json

Result: {"spec": <GP spec with optimised length_scales + variance>}.
"""
import argparse
import json
import sys

from doe.dataset import assemble
from doe.gp.surrogate import dataset_xy, observable_of, optimise_hyperparameters


def main():
  p = argparse.ArgumentParser(description='Optimise GP hyper-parameters to data.')
  p.add_argument('--model', required=True, metavar='FILE', help='Base GP spec JSON (kernel/noise/observables).')
  p.add_argument('--batches', required=True, nargs='+', metavar='FILE', help='Batch JSON files.')
  p.add_argument('--output', metavar='FILE', default=None, help='Write JSON results to FILE.')
  args = p.parse_args()

  with open(args.model) as f:
    spec = json.load(f)
  observable = observable_of(spec)  # asserts a single observable
  X, y = dataset_xy(assemble(args.batches), observable)
  if len(y) == 0:
    print(f'error: no data measures observable {observable!r}', file=sys.stderr)
    sys.exit(1)

  result = {'spec': optimise_hyperparameters(spec, X, y)}

  output = json.dumps(result, indent=2)
  if args.output is None:
    print(output)
  else:
    with open(args.output, 'w') as f:
      f.write(output)


if __name__ == '__main__':
  main()

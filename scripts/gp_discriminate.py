"""
Discriminative design of experiments for a fitted GP: choose a batch that best resolves
the sign pattern [f(x) > threshold] over a grid (see doe.doe.discriminative). Store-free.

Usage:
  python scripts/gp_discriminate.py --model SPEC.json --state STATE.json --grid GRID.json \
      --threshold T --bounds BOUNDS.json --batch-size N [--seed S] --output RESULT.json

GRID.json is a list of [A, B, E, temperature, t] rows; BOUNDS.json a list of [lo, hi] per dim.
Result: a design record body {"experiments": {...}, "auxiliary": {"expected": {...}, "eig": float}}.
"""
import argparse
import json

from doe.gp.surrogate import design_body, discriminatory_doe


def main():
  p = argparse.ArgumentParser(description='Discriminative GP DoE.')
  p.add_argument('--model', required=True, metavar='FILE', help='GP spec JSON.')
  p.add_argument('--state', required=True, metavar='FILE', help='Serialised GP state JSON.')
  p.add_argument('--grid', required=True, metavar='FILE', help='JSON list of grid-point rows.')
  p.add_argument('--threshold', type=float, required=True, metavar='T')
  p.add_argument('--bounds', required=True, metavar='FILE', help='JSON list of [lo, hi] per input dim.')
  p.add_argument('--batch-size', type=int, required=True, metavar='N', dest='batch_size')
  p.add_argument('--seed', type=int, default=0, metavar='S')
  p.add_argument('--output', metavar='FILE', default=None, help='Write JSON results to FILE.')
  args = p.parse_args()

  with open(args.model) as f:
    spec = json.load(f)
  with open(args.state) as f:
    state = json.load(f)
  with open(args.grid) as f:
    grid = json.load(f)
  with open(args.bounds) as f:
    bounds = [tuple(b) for b in json.load(f)]

  B, eig = discriminatory_doe(spec, state, grid, args.threshold, bounds, args.batch_size, seed=args.seed)
  result = design_body(spec, state, B, eig)

  output = json.dumps(result, indent=2)
  if args.output is None:
    print(output)
  else:
    with open(args.output, 'w') as f:
      f.write(output)


if __name__ == '__main__':
  main()

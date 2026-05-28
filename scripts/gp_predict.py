"""
Evaluate a fitted GP at explicit input points. Store-free; file in / result out.

Usage:
  python scripts/gp_predict.py --model SPEC.json --state STATE.json --points POINTS.json --output RESULT.json

POINTS.json is a list of [A, B, E, temperature, t] rows. Result: {"<observable>": {mean, std}}.
"""
import argparse
import json

from doe.gp.surrogate import predict_surrogate


def main():
  p = argparse.ArgumentParser(description='Predict with a fitted GP at explicit points.')
  p.add_argument('--model', required=True, metavar='FILE', help='GP spec JSON.')
  p.add_argument('--state', required=True, metavar='FILE', help='Serialised GP state JSON.')
  p.add_argument('--points', required=True, metavar='FILE', help='JSON list of input-point rows.')
  p.add_argument('--output', metavar='FILE', default=None, help='Write JSON results to FILE.')
  args = p.parse_args()

  with open(args.model) as f:
    spec = json.load(f)
  with open(args.state) as f:
    state = json.load(f)
  with open(args.points) as f:
    points = json.load(f)

  result = predict_surrogate(spec, state, points)

  output = json.dumps(result, indent=2)
  if args.output is None:
    print(output)
  else:
    with open(args.output, 'w') as f:
      f.write(output)


if __name__ == '__main__':
  main()

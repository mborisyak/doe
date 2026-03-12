import os
import numpy as np

def main():
  import argparse

  parser = argparse.ArgumentParser(description='Enzyme kinetics experiment')
  parser.add_argument('--seed', type=int, required=True, help='Seed for parameter generation')
  parser.add_argument('--config', required=False, default=None, help='Path to the config file')
  parser.add_argument('--output', required=True, help='Path to the output file')
  args = parser.parse_args()

  seed = args.seed

  if args.config is None:
    # <root> / scripts / experiment.py
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root, 'config', 'config.yaml')
  else:
    config_path = args.config

  with open(config_path, 'r') as f:
    import yaml
    config = yaml.safe_load(f)

  bounds = config['parameters']
  rng = np.random.default_rng(seed)

  parameters = {
    name: float(rng.uniform(low=low, high=high, size=()))
    for name, (low, high) in bounds.items()
  }

  with open(args.output, 'w') as f:
    import yaml
    yaml.safe_dump(parameters, f)

if __name__ == '__main__':
  main()
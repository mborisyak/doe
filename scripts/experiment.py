import os
import numpy as np

import jax

import matplotlib.pyplot as plt

import enzyme

def get_device(device_name):
  if ':' in device_name:
    platform, index = device_name.split(':')
    index = int(index)
  else:
    platform = device_name
    index = 0

  return jax.devices(platform)[index]

def get_rhs(parameters, device):
  def rhs(_, A, A0, B0, E, temp):
    delta = A0 - A
    B = B0 - delta
    C = delta
    D = delta

    rate = enzyme.kinetics(A, B, C, D, E, temp, parameters)

    return -rate

  rhs = jax.jit(rhs, device=device)
  jac = jax.jit(jax.jacobian(rhs, argnums=1), device=device)

  return rhs, jac

def simulate():
  pass

def main():
  import argparse
  from scipy.integrate import solve_ivp

  parser = argparse.ArgumentParser(description='Enzyme kinetics experiment')
  parser.add_argument('--parameters', required=True, help='Path to the parameter file')
  parser.add_argument('--seed', type=int, required=False, default=None, help='Seed for noise generation')
  parser.add_argument('--output', required=True, help='Path to the output file')
  parser.add_argument('--conditions', required=True, help='JSON file with initial conditions')
  parser.add_argument('--device', required=False, default='cpu', help='Number of measurements')
  parser.add_argument('--config', required=False, default=None, help='Path to the config file')
  parser.add_argument(
    '--plot', required=False, default=None,
    help='Path to the plot of the results, if not specified the plot is not going to be produced. '
         'The plot is for debug purposes only.'
  )
  args = parser.parse_args()

  if args.config is None:
    # <root> / scripts / experiment.py
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root, 'config', 'config.yaml')
  else:
    config_path = args.config

  with open(config_path, 'r') as f:
    import yaml
    config = yaml.safe_load(f)

  with open(args.parameters, 'r') as f:
    import yaml
    parameters = yaml.safe_load(f)
    parameters = {
      name: np.atleast_1d(np.array(p, dtype=np.float32))
      for name, p in parameters.items()
    }

  device = get_device(args.device)
  rhs, jac = get_rhs(parameters, device)

  T = config['experiment']['duration']
  n = config['experiment']['measurements']
  ts_eval = np.linspace(0, T, num=n + 2, dtype=np.float32)[1:-1]

  with open(args.conditions, 'r') as f:
    import json
    conditions = json.load(f)

  results = dict()
  trajectories = dict()

  rng = np.random.default_rng(args.seed)
  sigma = config['noise']

  ### concentrations
  Ac = config['solutions']['A']
  Bc = config['solutions']['B']
  Ec = config['solutions']['E']

  convert = lambda x: np.atleast_1d(np.asarray(x, dtype=np.float32))

  for label, experiment in conditions.items():
    A0_volume = convert(experiment['A'])
    B0_volume = convert(experiment['B'])
    E_volume = convert(experiment['E'])

    V = A0_volume + B0_volume + E_volume
    A0 = A0_volume * Ac / V
    B0 = B0_volume * Bc / V
    E = E_volume * Ec / V

    temperature = convert(experiment['temperature'])

    trajectory = solve_ivp(
      rhs, t_span=(0.0, T), t_eval=ts_eval, y0=A0,
      args=(A0, B0, E, temperature), jac=jac, method='LSODA'
    )
    timestamps = trajectory.t
    y = trajectory.y[0]
    measurements = y + sigma * rng.normal(size=y.shape).astype(np.float32)

    results[label] = {
      'timestamps': [float(t) for t in timestamps],
      'measurements': [float(y) for y in measurements]
    }

    if args.plot is not None:
      trajectory = solve_ivp(
        rhs, t_span=(0.0, T), t_eval=np.linspace(0.0, T, num=int(128 * T)), y0=A0,
        args=(A0, B0, E, temperature), jac=jac, method='LSODA'
      )
      trajectories[label] = (trajectory.t, trajectory.y[0])

  with open(args.output, 'w') as f:
    import json
    json.dump(results, f, indent=2)

  if args.plot is not None:
    fig = plt.figure(figsize=(9, 6))
    axes = fig.subplots(1, 1)

    for i, label in enumerate(trajectories):
      ts, ys = trajectories[label]
      A, B, E = conditions[label]['A'], conditions[label]['B'], conditions[label]['E']
      temperature = conditions[label]['temperature']
      description = f'{label}: A={A}, B={B}, E={E}, T={temperature} C'

      axes.plot(ts, ys, label=description, color=plt.cm.tab10(i))
      axes.scatter(results[label]['timestamps'], results[label]['measurements'], color=plt.cm.tab10(i))

    axes.legend()
    axes.set_xlabel('time, sec')
    axes.set_ylabel('concentration, mM')
    fig.tight_layout()
    fig.savefig(args.plot)
    plt.close(fig)

if __name__ == '__main__':
  main()
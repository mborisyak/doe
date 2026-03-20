import os
import json
from collections import namedtuple

import numpy as np

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import doe
from doe.common import Conditions


def test_mle(plot_root, seed):
  root = os.path.dirname(os.path.dirname(__file__))
  data_path = os.path.join(root, 'data')

  with open(os.path.join(data_path, 'models', 'super-model-2.json'), 'r') as f:
    model_spec = json.load(f)

  model = doe.common.CustomODESystem(model_spec)

  with open(os.path.join(data_path, 'experiments', 'example.json'), 'r') as f:
    conditions = json.load(f)

  with open(os.path.join(data_path, 'experiments', 'measurements.json'), 'r') as f:
    measurements = json.load(f)

  condition_ranges = Conditions(A=[0.1, 5.0], B=[0.1, 5.0], E=[0.1, 5.0], temperature=[0.0, 100.0])

  parameter_ranges = model.parameter_ranges()
  parameters = model.Parameters(**{
    f: (low + high) / 2
    for f, (low, high) in zip(parameter_ranges._fields, parameter_ranges)
  })

  test_label, *_ = measurements
  trajectory = model.solve(
    doe.common.Conditions(**conditions[test_label]),
    jnp.array(measurements[test_label]['timestamps']),
    parameters
  )

  plt.figure()
  plt.plot(jnp.array(measurements[test_label]['timestamps']), trajectory)
  plt.savefig(os.path.join(plot_root, 'test.png'))
  plt.close()

  # estimate parameters from existing data
  losses, parameters, predictions = doe.inference.maximum_likelihood_estimate(
    model, conditions, measurements, model.parameter_ranges(), iterations=None, mode='euler'
  )

  print(np.diff(losses) / (losses[0] - losses[1:]))

  n = len(measurements)
  fig, axes = plt.subplots(n + 1, 1, figsize=(8, 5 * n))

  axes[0].plot(losses)
  axes[0].set_title('learning curve')

  for i, label in enumerate(measurements):
    axes[i + 1].scatter(measurements[label]['timestamps'], measurements[label]['measurements'], color='black')
    axes[i + 1].scatter(measurements[label]['timestamps'], predictions[label])
    axes[i + 1].set_title(f'Experiment {label}')

  fig.tight_layout()
  fig.savefig(os.path.join(plot_root, 'mle.png'))
  plt.close(fig)

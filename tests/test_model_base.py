import os
from collections import namedtuple

import numpy as np

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import doe
from doe.common import Conditions

Parameters = namedtuple('Parameters', ['q', 'K_A', 'K_B'])

class SimpleEnzyme(doe.common.ODEModel[Parameters]):
  def initial_state(self, conditions: Conditions) -> jax.Array:
    Ac, Bc, Ec = doe.common.get_initial_concentrations(conditions)

    return jnp.stack([Ac, Bc, Ec], axis=-1)

  def rhs(self, state: jax.Array, conditions: Conditions, parameters: Parameters) -> jax.Array:
    A, B, E = state[..., 0], state[..., 1], state[..., 2]
    rate = E * parameters.q * A / (parameters.K_A + A) * B / (B + parameters.K_B)

    return jnp.stack([-rate, -rate, jnp.zeros_like(rate)], axis=-1)

  def observables(self, state: jax.Array) -> jax.Array:
    return state[..., 0]

def test_base_model(plot_root):
  root = os.path.dirname(__file__)

  with open(os.path.join(root, 'example.json'), 'r') as f:
    import json
    conditions = json.load(f)

  with open(os.path.join(root, 'measurements.json'), 'r') as f:
    import json
    measurements = json.load(f)

  simple_enzyme = SimpleEnzyme()
  parameter_ranges = Parameters(q=[1.0e+2, 2.0e+3], K_A=[1.0e-2, 2.0], K_B=[1.0e-2, 2.0])

  losses, parameters, predictions = doe.inference.maximum_likelihood_estimate(
    simple_enzyme, conditions, measurements, parameter_ranges, iterations=32
  )

  n = len(conditions) + 1
  fig = plt.figure(figsize=(6, 4 * n))
  axes = fig.subplots(n, )

  axes[0].plot(losses)
  for i, label in enumerate(conditions):
    axes[i + 1].scatter(measurements[label]['timestamps'], measurements[label]['measurements'], label=label)
    axes[i + 1].plot(measurements[label]['timestamps'], predictions[label])
    axes[i + 1].legend(loc='upper right')
  fig.tight_layout()
  fig.savefig(os.path.join(plot_root, 'simple.png'))
  plt.close(fig)
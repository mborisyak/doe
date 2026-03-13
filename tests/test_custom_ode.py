import os
import json
from collections import namedtuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import doe
from doe.common import Conditions


def test_custom_ode(plot_root, seed):
  root = os.path.dirname(os.path.dirname(__file__))

  with open(os.path.join(root, 'data', 'models', 'simple.json'), 'r') as f:
    spec = json.load(f)

  model = doe.common.CustomODESystem(spec)

  with open(os.path.join(root, 'data', 'experiments', 'example.json'), 'r') as f:
    conditions_data = json.load(f)

  with open(os.path.join(root, 'data', 'experiments', 'measurements.json'), 'r') as f:
    measurements_data = json.load(f)

  labels = [k for k in conditions_data]
  selection = labels[:2]

  conditions_data = {k: conditions_data[k] for k in selection}
  measurements_data = {k: measurements_data[k] for k in selection}

  parameter_ranges = model.parameter_ranges()
  condition_ranges = Conditions(A=[0.1, 5.0], B=[0.1, 5.0], E=[0.1, 5.0], temperature=[0.0, 100.0])

  parameters = model.Parameters(q=1.0e+3, K_A=1.0, K_B=1.0)
  conditions = doe.common.Conditions(A=2.0, B=2.0, E=2.0, temperature=37.0)

  state = model.initial_state(conditions)
  print(state)

  ds_dt = model.rhs(state, conditions, parameters)
  print(ds_dt)
  assert jnp.allclose(ds_dt, jnp.array([-0.25, -0.25, 0.0]))

  trajectory = model.solve(conditions, jnp.linspace(0, 1, num=10), parameters)

  # estimate parameters from existing data
  _, parameters, _ = doe.inference.maximum_likelihood_estimate(
    model, conditions_data, measurements_data, parameter_ranges, iterations=512
  )

  fisher = doe.doe.Fisher(
    model, condition_ranges, parameter_ranges,
    timestamps=jnp.linspace(1.0, 29.0, 9),
    iterations=64,
  )

  # prepare encoded history
  label_order = list(conditions_data.keys())
  dtype = jnp.float32

  controls = jnp.stack([
    fisher.encode_conditions(Conditions(**{
      k: jnp.array(conditions_data[label][k], dtype=dtype)
      for k in Conditions._fields
    }))
    for label in label_order
  ])  # (n_e, n_cond)

  timestamps = jnp.array(
    [measurements_data[label]['timestamps'] for label in label_order], dtype=dtype
  )  # (n_e, n_t)

  # propose a batch of 3 experiments
  n = 3
  key = jax.random.PRNGKey(seed)
  loss_trace, proposed = fisher.propose(key, n=n, controls=controls, timestamps=timestamps, parameters=parameters)

  assert proposed.shape == (n, len(Conditions._fields))
  assert jnp.all(jnp.isfinite(proposed))
  assert jnp.all(jnp.isfinite(loss_trace))

  # plot
  fields = Conditions._fields
  fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))

  for i in range(n):
    proposed_conditions = fisher.decode_conditions(proposed[i])
    axes[1, i].bar(fields, [float(getattr(proposed_conditions, f)) for f in fields])
    axes[1, i].set_title(f'Proposed conditions {i + 1}')

  axes[0, 0].plot(loss_trace)
  axes[0, 0].set_title('Fisher loss')
  axes[0, 0].set_xlabel('iteration')
  for i in range(1, n):
    axes[0, i].set_visible(False)

  fig.tight_layout()
  fig.savefig(os.path.join(plot_root, 'fisher.png'))
  plt.close(fig)
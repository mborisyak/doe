import os
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# ODE helpers
# -----------------------------------------------------------------------

State = namedtuple('State', ['x', 'y'])
Parameters = namedtuple('Parameters', ['alpha', 'beta', 'a', 'b'])

def test_clip_vs_jacrev_jacfwd(plot_root):
  dt = 1.0e-2
  n = 1000

  def f(state, parameters):
    x, y = state
    alpha, beta, a, b = parameters
    return State(alpha * y + a, beta * x + b)

  def euler(initial_state, parameters):
    def step(state, _):
      dstate_dt = f(state, parameters)
      clipped = State(
        x=jnp.clip(state.x + dt * dstate_dt.x, 0),
        y=state.y + dt * dstate_dt.y,
      )
      return clipped, clipped

    _, trajectory = jax.lax.scan(step, init=initial_state, length=n)
    return trajectory

  def loss(initial_state, parameters):
    trajectory = euler(initial_state, parameters)
    return jnp.mean(jnp.square(trajectory.x) + jnp.square(trajectory.y))

  grad_fwd = jax.jacfwd(loss, argnums=1)
  grad_rev = jax.jacrev(loss, argnums=1)

  state0 = State(1.0, 0.0)
  parameters = Parameters(1.0, -1.0, -0.5, 0.5)

  solution = euler(state0, parameters)

  print()
  print(grad_fwd(state0, parameters))
  print(grad_rev(state0, parameters))

  fig = plt.figure(figsize=(8, 4))
  axes = fig.subplots(1, 2)
  axes[0].plot(solution.x)
  axes[0].plot(solution.y)
  axes[1].plot(solution.x, solution.y)
  fig.savefig(os.path.join(plot_root, 'clipped.png'))
  plt.close(fig)
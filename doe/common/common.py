from typing import TypeVar, Generic
from collections import namedtuple

import jax
import jax.numpy as jnp

import diffrax

SOLUTIONS = {
  'A': 3.0,
  'B': 3.0,
  'E': 3.0e-3,
}

NOISE_SIGMA = 0.025

TIME_HORIZON = 30
MEASUREMENT_TIMESTAMPS = jnp.linspace(0, 30, num=11)[1:-1]

Conditions = namedtuple('Conditions', ['A', 'B', 'E', 'temperature'])

### should be a named tuple
Parameters = TypeVar('Parameters')

def get_initial_concentrations(condition):
  A, B, E, _ = condition
  V_total = A + B + E

  Ac = A * SOLUTIONS['A'] / V_total
  Bc = B * SOLUTIONS['B'] / V_total
  Ec = E * SOLUTIONS['E'] / V_total

  return Ac, Bc, Ec

class ODEModel(Generic[Parameters]):
  def __init__(self):
    def rhs(_, state, args):
      condition, parameters = args
      return self.rhs(state, condition, parameters)

    self.term = diffrax.ODETerm(rhs)
    self.solver = diffrax.Tsit5()

  def initial_state(self, conditions: Conditions) -> jax.Array:
    """
    Returns initial state of the system given conditions.
    Conditions typically contain initial states of the

    :param conditions: initial conditions of the experiment
    :return: initial state of the system.
    """
    raise NotImplementedError()

  def rhs(self, state: jax.Array, conditions: Conditions, parameters: Parameters) -> jax.Array:
    raise NotImplementedError()

  def observables(self, state: jax.Array, parameters: Parameters) -> jax.Array:
    raise NotImplementedError()

  def trajectory(self, conditions, timestamps, parameters):
    result = diffrax.diffeqsolve(
      self.term, self.solver,
      dt0=0.1, t0=0, t1=TIME_HORIZON, saveat=diffrax.SaveAt(ts=timestamps),
      y0=self.initial_state(conditions),
      args=(conditions, parameters),
      stepsize_controller=diffrax.ConstantStepSize(),
      max_steps=int(TIME_HORIZON / 0.1) + 1,
      adjoint=diffrax.DirectAdjoint(),
    )

    return result.ys

  def solve(self, conditions, timestamps, parameters):
    states = self.trajectory(conditions, timestamps, parameters)
    return self.observables(states, parameters)

  def parameter_ranges(self) -> Parameters:
    raise NotImplementedError()
from collections import namedtuple

import jax
import sympy as sp

import jax.numpy as jnp

from . import Parameters, Conditions
from .common import ODEModel, get_initial_concentrations

__all__ = [
  'CustomODESystem'
]

def compile_cse(inputs, replacements, outputs):
  compiled_steps = []

  for sym, expr in replacements.items():
    deps = list(expr.free_symbols)
    func = sp.lambdify(deps, expr, "jax")
    compiled_steps.append((sym, deps, func))

  compiled_outputs = []
  for expr in outputs:
    deps = list(expr.free_symbols)
    func = sp.lambdify(deps, expr, "jax")
    compiled_outputs.append((deps, func))

  def f(*args):
    env = dict(zip(inputs, args))

    for sym, deps, func in compiled_steps:
      env[sym] = func(*[env[d] for d in deps])

    return [func(*[env[d] for d in deps]) for deps, func in compiled_outputs]

  return f

CONDITIONS = ['A0', 'B0', 'E0', 'T']

class CustomODESystem(ODEModel):
  def __init__(self, spec):
    self.states = spec['states']
    self.parameters = spec['parameters']

    self.Parameters = namedtuple('Parameters', self.parameters.keys())
    self._parameter_rages = self.Parameters(**{
      name: (low, high) for name, (low, high) in self.parameters.items()
    })

    states = {name: sp.Symbol(name) for name in spec['states']}
    parameters = {name: sp.Symbol(name) for name in spec['parameters']}
    conditions = {name: sp.Symbol(name) for name in CONDITIONS}

    algebraics = {}
    expressions_algebraics = {}
    for name, definition in spec['algebraics'].items():
      sym = sp.Symbol(name)
      algebraics[name] = sym
      expressions_algebraics[sym] = sp.parse_expr(definition, local_dict={**algebraics, **states, **conditions, **parameters})

    expressions_rhs = {
      k: sp.parse_expr(expr, local_dict={**algebraics, **states, **parameters})
      for k, expr in spec['rhs'].items()
    }

    self._rhs_compiled = compile_cse(
      inputs=[*states.values(), *conditions.values(), *parameters.values()],
      replacements=expressions_algebraics,
      outputs=[*expressions_rhs.values()]
    )

    for s in states:
      assert s in spec['initial_state'], f'State {s} is missing from initial state declaration'

    for s in spec['initial_state']:
      assert s in states, f'State {s} is declared in `initial_state` but not in `states`.'

    initial_rhs = {
      k: sp.parse_expr(expr, local_dict=conditions)
      for k, expr in spec['initial_state'].items()
    }

    self._initial_compiled = compile_cse(
      inputs=conditions.values(),
      replacements={},
      outputs=[initial_rhs[s] for s in states]
    )

    assert tuple(spec['observables']) == ('A', )

    obs_expr = sp.parse_expr(spec['observables']['A'], local_dict=states)
    self._observables_compiled = sp.lambdify(states.values(), obs_expr, "jax")
    
    super().__init__()

  def rhs(self, state, conditions, parameters):
    named_states = [state[..., i] for i, _ in enumerate(self.states)]
    named_conditions = [conditions.A, conditions.B, conditions.E, conditions.temperature]
    named_parameters = [getattr(parameters, name) for name in self.parameters]

    return jnp.stack(self._rhs_compiled(*named_states, *named_conditions, *named_parameters), axis=-1)

  def parameter_ranges(self) -> Parameters:
    return self._parameter_rages

  def initial_state(self, conditions: Conditions) -> jax.Array:
    Ac, Bc, Ec = get_initial_concentrations(conditions)
    initial_state = self._initial_compiled(Ac, Bc, Ec, conditions.temperature)
    return jnp.stack(initial_state, axis=-1)

  def observables(self, state):
    named_states = [state[..., i] for i, _ in enumerate(self.states)]
    obs = self._observables_compiled(*named_states)
    return obs
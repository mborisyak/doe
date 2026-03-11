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


def validate_spec(spec):
  groups = {
    'states': set(spec['states']),
    'parameters': set(spec['parameters']),
    'algebraics': set(spec.get('algebraics', {})),
  }
  pairs = [
    ('states', 'parameters'),
    ('states', 'algebraics'),
    ('parameters', 'algebraics'),
  ]
  for name_a, name_b in pairs:
    overlap = groups[name_a] & groups[name_b]
    if overlap:
      raise ValueError(f"Variable name conflict between '{name_a}' and '{name_b}': {sorted(overlap)}")


class CustomODESystem(ODEModel):
  """
  Converts serializable description of a system into an instance of ODEModel.

  spec is a dictionary with the following fields:
  - states: model fields, a list of strings;
  - parameters: model parameters with their ranges, a dictionary with parameter names as keys and
    a 2-element sequence of floats (lower, upper) as value.
  - initial_state: defines initial_state method, a dictionary: state name -> expression;
    The expression can use predefined condition variables: A0, B0, E0, T (temperature).
  - algebraics: a dictionary with definitions of temporary variables (algebraics) for rhs,
    variable name -> expression. The expressions can use previously defined variables, states, initial condition variables and parameters.
  - rhs: a dictionary with definitions of states' time derivatives, state name -> expression,
    the expressions can use algebraic variables (defined by "algebraics"), states, initial condition variables and parameters.
    Expressions in "rhs" can't use other time derivatives, algebraics can be used instead:
        "algebraics": {
          "rate": <expression>
        },
        "rhs": {
          "A": "-rate",
          "B": "-rate",
        }
  - observables: a dictionary with definitions of observables (currently only "A"). The expressions in "observables"
      can use state variables and parameter values. Often definitions are trivial, e.g.:
        "observables": {
          "A": "A"
        }
      assigns observable "A" the value of state "A".
  """
  def __init__(self, spec):
    validate_spec(spec)

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

    obs_expr = sp.parse_expr(spec['observables']['A'], local_dict={**states, **parameters})
    self._observables_compiled = sp.lambdify([*states.values(), *parameters.values()], obs_expr, "jax")
    
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

  def observables(self, state, parameters):
    named_states = [state[..., i] for i, _ in enumerate(self.states)]
    named_parameters = [getattr(parameters, name) for name in self.parameters]
    obs = self._observables_compiled(*named_states, *named_parameters)
    return obs
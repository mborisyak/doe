from ..common import ODEModel, Parameters, Conditions

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import optax

__all__ = [
  'maximum_likelihood_estimate'
]

default_optimizer = optax.lbfgs()

def stable_logit(xs):
  eps = jnp.finfo(xs.dtype).eps
  us = jnp.clip(xs, eps, 1 - eps)
  return jsp.special.logit(us)

@jax.jit
def encode(params: Parameters, param_ranges: Parameters) -> jax.Array:
  return jnp.stack([
    stable_logit(
      (p - low) / (high - low)
    )

    for p, (low, high) in zip(params, param_ranges)
  ], axis=0)

@jax.jit
def decode(params_encoded: jax.Array, param_ranges: Parameters) -> Parameters:
  return type(param_ranges)(*(
    jax.nn.sigmoid(params_encoded[j]) * (high - low) + low

    for j, (low, high) in enumerate(param_ranges)
  ))

def maximum_likelihood_estimate(
    model: ODEModel[Parameters], conditions, data,
    parameter_ranges: Parameters,
    initial=None, optimizer=default_optimizer,
    iterations: int | None=1024, rtol: float | None=1.0e-6,
    dtype=jnp.float32,
):
  """
  Returns maximum likelihood fit to the data.

  :param model: kinetic model;
  :param conditions: a dict label -> condition, each 'condition' is a dictionary
    with the following values: 'A', 'B', 'E', 'temperature' (as accepted by the simulation).
  :param data: label -> experiment, each experiment is a dictionary with 'timestamps' and 'measurement' keys.
  :param parameter_ranges: bounds on the parameters of the model;
  :param initial: if provided, an initial guess for the optimizer, middle of ranges if unspecified;
  :param dtype: dtype of computations
  :param optimizer: optax optimizer (or compatible), LBFGS by defult
  :param iterations: number of iterations, if rtol specified, the maximal number of iterations
  :param rtol: relative tolerance of stopping criterium.
  """

  def solve(cond, ts, params):
    return jax.vmap(model.solve, in_axes=(0, 0, None), out_axes=0)(cond, ts, params)

  @jax.jit
  def loss_fn(params_encoded, cond, ts, obs):
    params = decode(params_encoded, parameter_ranges)
    solution = solve(cond, ts, params)
    return jnp.mean(jnp.square(solution - obs))

  @jax.jit
  def value_grad(params_encoded, cond, ts, obs):
    return jax.value_and_grad(loss_fn, argnums=0)(params_encoded, cond, ts, obs)

  def step(params_encoded, cond, ts, obs, opt_state):
    loss, grad = value_grad(params_encoded, cond, ts, obs)
    updates, opt_state_updated = optimizer.update(
      grad, opt_state, params_encoded, value=loss, grad=grad, value_fn=lambda p: loss_fn(p, cond, ts, obs)
    )
    params_encoded_updated = optax.apply_updates(params_encoded, updates)

    return loss, params_encoded_updated, opt_state_updated

  label_order = [k for k in conditions]

  timestamps = jnp.array([
    data[label]['timestamps'] for label in label_order
  ], dtype=dtype)

  measurements = jnp.array([
    data[label]['measurements'] for label in label_order
  ], dtype=dtype)

  conditions = Conditions(**{
    k: jnp.array([conditions[label][k] for label in label_order], dtype=dtype)
    for k in Conditions._fields
  })

  if initial is None:
    parameters_encoded = np.zeros(shape=(len(parameter_ranges), ), dtype=dtype)
  else:
    parameters_encoded = encode(initial, parameter_ranges)

  state = optimizer.init(parameters_encoded)

  if iterations is None:
    if rtol is None:
      raise ValueError('either rtol, iterations or both must be specified')

    losses = list()

    l0, parameters_encoded, state = step(parameters_encoded, conditions, timestamps, measurements, state)
    losses.append(l0)

    while True:
      l, parameters_encoded, state = step(parameters_encoded, conditions, timestamps, measurements, state)

      if l[-1] - l < rtol * (l0 - l):
        losses.append(l)
        losses = np.array(losses)
        break
  else:
    losses = np.ndarray(shape=(iterations, ))

    losses[0], parameters_encoded, state = step(parameters_encoded, conditions, timestamps, measurements, state)
    for i in range(1, iterations):
      losses[i], parameters_encoded, state = step(parameters_encoded, conditions, timestamps, measurements, state)

      if rtol is not None:
        if losses[i - 1] - losses[i] < rtol * (losses[0] - losses[i]):
          losses = losses[:(i + 1)]
          break

  parameters = decode(parameters_encoded, parameter_ranges)
  predictions = solve(conditions, timestamps, parameters)
  predictions = {
    label: predictions[i]
    for i, label in enumerate(label_order)
  }

  return losses, parameters, predictions

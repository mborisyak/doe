from ..common import ODEModel, Parameters, Conditions

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import optax

__all__ = [
  'maximum_likelihood_estimate',
  'maximum_likelihood_estimate_euler',
  'maximum_likelihood_estimate_scipy',
]

default_optimizer = optax.adamw(learning_rate=1.0e-2, nesterov=True)

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

@jax.jit
def decode_vjp(cotangent, params_encoded, parameter_ranges):
  _, vjp_fn = jax.vjp(
    lambda pe: jnp.stack(decode(pe, parameter_ranges), axis=-1),
    params_encoded
  )
  grad_encoded = vjp_fn(cotangent)[0]
  return grad_encoded

@jax.jit
def vmap_decode_vjp(cotangent, params_encoded, parameter_ranges):
  return jax.vmap(decode_vjp, in_axes=(0, None, None))(cotangent, params_encoded, parameter_ranges)

def maximum_likelihood_estimate_euler(
    model: ODEModel[Parameters], conditions, data,
    parameter_ranges: Parameters,
    initial=None, optimizer=default_optimizer,
    iterations: int | None=None, rtol: float | None=1.0e-6,
    dtype=jnp.float32, error_tol: float | None=1.0e-3, device=None
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
  :param error_tol: tolerance for the integration error
  :param device: compilation target.
  """

  label_order = [k for k in conditions]

  timestamps = jnp.array([
    data[label]['timestamps'] for label in label_order
  ], dtype=dtype)

  measurements = jnp.array([
    data[label]['measurements'] for label in label_order
  ], dtype=dtype)

  conditions_batched = Conditions(**{
    k: jnp.array([conditions[label][k] for label in label_order], dtype=dtype)
    for k in Conditions._fields
  })

  if initial is None:
    parameters_encoded = np.zeros(shape=(len(parameter_ranges), ), dtype=dtype)
  else:
    parameters_encoded = encode(initial, parameter_ranges)

  def loss_fn(ps_encoded, cs, ts, ys):
    ps = decode(ps_encoded, parameter_ranges)
    pred, errs = jax.vmap(model.solve_euler, in_axes=(0, 0, None))(cs, ts, ps)
    err = jnp.mean(errs)
    return jnp.mean(jnp.square(pred - ys)), err

  def step(ps_encoded, cs, ts, ys, opt_state):
    (ls, err), grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
      ps_encoded, cs, ts, ys
    )

    updates, opt_state_updated = optimizer.update(
      grad, opt_state, ps_encoded, value=ls, grad=grad
    )
    params_encoded_updated = optax.apply_updates(ps_encoded, updates)
    return ls, err, params_encoded_updated, opt_state_updated

  step = jax.jit(step, device=device)
  state = optimizer.init(parameters_encoded)

  if iterations is None:
    if rtol is None:
      raise ValueError('either rtol, iterations or both must be specified')

    losses = list()

    l0, err, parameters_encoded, state = step(parameters_encoded, conditions_batched, timestamps, measurements, state)
    losses.append(l0)
    assert error_tol is None or (jnp.isfinite(err) and err < error_tol), \
      f'step 0: Integration error is too large! {err} / {error_tol}'

    while True:
      loss, err, parameters_encoded, state = step(parameters_encoded, conditions_batched, timestamps, measurements, state)
      assert error_tol is None or (jnp.isfinite(err) and err < error_tol), \
        f'step {len(losses) - 1}: Integration error is too large! {err} / {error_tol}'

      improvement = losses[-1] - loss
      baseline = rtol * (l0 - loss)
      losses.append(loss)

      if improvement < baseline:
        losses = np.array(losses)
        break

  else:
    losses = np.ndarray(shape=(iterations, ))

    losses[0], err, parameters_encoded, state = step(parameters_encoded, conditions_batched, timestamps, measurements, state)
    assert error_tol is None or (jnp.isfinite(err) and err < error_tol), \
      f'step 0: Integration error is too large! {err} / {error_tol}'

    for i in range(1, iterations):
      losses[i], err, parameters_encoded, state = step(parameters_encoded, conditions_batched, timestamps, measurements, state)
      assert error_tol is None or (jnp.isfinite(err) and err < error_tol), \
        f'step {i}: Integration error is too large! {err} / {error_tol}'

      if rtol is not None:
        if losses[i - 1] - losses[i] < rtol * (losses[0] - losses[i]):
          losses = losses[:(i + 1)]
          break

  parameters = decode(parameters_encoded, parameter_ranges)

  predictions = {}
  for i, label in enumerate(label_order):
    cond_i = type(conditions_batched)(*(field[i] for field in conditions_batched))
    predictions[label] = model.solve(cond_i, timestamps[i], parameters)

  return losses, parameters, predictions

def maximum_likelihood_estimate_scipy(
    model: ODEModel[Parameters], conditions, data,
    parameter_ranges: Parameters,
    initial=None, method='L-BFGS-B',
    iterations: int | None=None, rtol: float | None=1.0e-8,
    dtype=jnp.float32
):
  """
  Returns maximum likelihood fit using scipy bounded optimizer (L-BFGS-B) and LSODA integrator.

  :param model: kinetic model;
  :param conditions: a dict label -> condition, each 'condition' is a dictionary
    with the following values: 'A', 'B', 'E', 'temperature'.
  :param data: label -> experiment, each experiment is a dictionary with 'timestamps' and 'measurements' keys.
  :param parameter_ranges: bounds on the parameters of the model;
  :param initial: if provided, an initial guess for the optimizer, middle of ranges if unspecified;
  :param method: scipy.optimize.minimize method, L-BFGS-B by default;
  :param iterations: max number of optimizer iterations;
  :param rtol: function value tolerance for convergence;
  :param dtype: dtype of computations.
  """
  from scipy.optimize import minimize

  label_order = [k for k in conditions]

  timestamps = jnp.array([
    data[label]['timestamps'] for label in label_order
  ], dtype=dtype)

  measurements = jnp.array([
    data[label]['measurements'] for label in label_order
  ], dtype=dtype)

  conditions_batched = Conditions(**{
    k: jnp.array([conditions[label][k] for label in label_order], dtype=dtype)
    for k in Conditions._fields
  })

  bounds = [(float(low), float(high)) for low, high in parameter_ranges]

  if initial is None:
    x0 = np.array([(float(low) + float(high)) / 2.0 for low, high in parameter_ranges])
  else:
    x0 = np.array(list(initial))

  mse = lambda obs, ys: jnp.mean(jnp.square(obs - ys))

  losses = []

  def objective(x):
    params = type(parameter_ranges)(*(x[i] for i in range(len(parameter_ranges))))
    values, dense_grad = model.accumulated_value_gradient_scipy(
      mse, conditions_batched, timestamps, params, arguments=measurements
    )
    loss = float(jnp.mean(values))
    grad = np.array(jnp.mean(dense_grad, axis=0))
    losses.append(loss)
    return loss, grad

  options = {}
  if iterations is not None:
    options['maxiter'] = iterations
  if rtol is not None:
    options['ftol'] = rtol

  result = minimize(objective, x0, method=method, jac=True, bounds=bounds, options=options)

  parameters = type(parameter_ranges)(*(result.x[i] for i in range(len(parameter_ranges))))

  predictions = {}
  for i, label in enumerate(label_order):
    cond_i = type(conditions_batched)(*(field[i] for field in conditions_batched))
    predictions[label] = model.solve(cond_i, timestamps[i], parameters)

  return np.array(losses), parameters, predictions


def maximum_likelihood_estimate(
    model: ODEModel[Parameters], conditions, data,
    parameter_ranges: Parameters,
    initial=None, optimizer=default_optimizer,
    iterations: int | None=None, rtol: float | None=1.0e-6,
    dtype=jnp.float32, mode='auto'
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
  :param optimizer: optax optimizer (or compatible), LBFGS by default
  :param iterations: number of iterations, if rtol specified, the maximal number of iterations
  :param rtol: relative tolerance of stopping criterium.
  :param mode: 'euler', 'scipy', or 'auto' (tries euler, falls back to scipy on integration error)
  """
  if mode == 'euler':
    return maximum_likelihood_estimate_euler(
      model, conditions, data, parameter_ranges,
      initial=initial, optimizer=optimizer, iterations=iterations, rtol=rtol,
      dtype=dtype
    )
  elif mode == 'scipy':
    return maximum_likelihood_estimate_scipy(
      model, conditions, data, parameter_ranges,
      initial=initial, iterations=iterations, rtol=rtol, dtype=dtype
    )
  elif mode == 'auto':
    try:
      return maximum_likelihood_estimate_euler(
        model, conditions, data, parameter_ranges,
        initial=initial, optimizer=optimizer, iterations=iterations, rtol=rtol,
        dtype=dtype
      )
    except AssertionError:
      return maximum_likelihood_estimate_scipy(
        model, conditions, data, parameter_ranges,
        initial=initial, iterations=iterations, rtol=rtol, dtype=dtype
      )
  else:
    raise ValueError(f'unknown mode: {mode!r}, expected euler, scipy, or auto')

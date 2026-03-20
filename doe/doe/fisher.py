import jax
import jax.numpy as jnp

from ..common import ODEModel, Conditions
from .line_search import ArmijoLineSearch

__all__ = [
  'Fisher'
]


class Fisher:
  EIGENVALUE_THRESHOLD = 1.0e-3

  def __init__(
      self, model: ODEModel, condition_ranges, parameter_ranges,
      timestamps: jax.Array,
      iterations: int, regularization: float | None = None, criterion: str = 'D',
      line_search: ArmijoLineSearch | None = None,
  ):
    self.model = model
    self.condition_ranges = condition_ranges
    self.parameter_ranges = parameter_ranges

    def decode_ranges(encoded, ranges):
      return type(ranges)(*(
        jax.nn.sigmoid(encoded[j]) * (high - low) + low
        for j, (low, high) in enumerate(ranges)
      ))

    def encode_ranges(values, ranges):
      return jnp.stack([
        jax.scipy.special.logit(jnp.clip((v - low) / (high - low), 1e-6, 1 - 1e-6))
        for v, (low, high) in zip(values, ranges)
      ])

    @jax.jit
    def estimate_raw(cond_encoded, ts, params_encoded):
      cond = decode_ranges(cond_encoded, condition_ranges)
      params = decode_ranges(params_encoded, parameter_ranges)
      trajectory, error = model._solve_euler(cond, ts, params)
      return trajectory.ravel(), error

    @jax.jit
    def jacobian(controls, ts, encoded_parameters):
      # controls: (n_e, n_cond), ts: (n_e, n_t), encoded_parameters: (n_params,)
      # output:   (n_e, n_t*n_obs, n_params)
      # jacfwd is used so the inner Jacobian (w.r.t. params) is computed via
      # forward-mode AD, which is composable with the outer reverse-mode gradient
      # (w.r.t. controls). Reverse-over-reverse through an adaptive ODE while_loop
      # is not supported by JAX/diffrax.
      return jax.vmap(
        jax.jacfwd(estimate_raw, argnums=2, has_aux=True),
        in_axes=(0, 0, None),
        out_axes=0,
      )(controls, ts, encoded_parameters)

    @jax.jit
    def information(controls, ts, encoded_parameters):
      J, error = jacobian(controls, ts, encoded_parameters)
      # J: (n_e, n_t*n_obs, n_params); contract over obs axis, batch over n_e
      I = jax.lax.dot_general(J, J, (([1], [1]), ([0], [0])))
      # I: (n_e, n_params, n_params); sum over experiments
      error = jnp.sum(error) / max(J.shape[0], 1)
      return jnp.sum(I, axis=0), error

    if criterion == 'D':
      @jax.jit
      def criterion_fn(I):
        svdvals = jnp.linalg.svdvals(I)
        return -2 * jnp.sum(jnp.log(svdvals))
    elif criterion == 'A':
      @jax.jit
      def criterion_fn(I):
        svdvals = jnp.linalg.svdvals(I)
        return jnp.sum(jnp.square(1 / svdvals))
    else:
      raise ValueError('criterion must be either A or D.')

    @jax.jit
    def loss(controls, ts, encoded_parameters, I0):
      I, error = information(controls, ts, encoded_parameters)
      l = criterion_fn(I + I0)
      if regularization is None:
        return l, error
      else:
        return l + regularization * jnp.sum(jnp.square(controls)), error

    if line_search is None:
      line_search = ArmijoLineSearch(x_min=-3.0, x_max=3.0)

    def _propose(initial, encoded_parameters, I0):
      n_e, _ = initial.shape
      ts = jnp.broadcast_to(timestamps[None], (n_e, timestamps.shape[0]))
      opt_state = line_search.init(initial)

      def value_fn(proposal):
        return loss(proposal, ts, encoded_parameters, I0)

      def _step(carry, _):
        proposal, state = carry
        (value, error), grads = jax.value_and_grad(value_fn, has_aux=True)(proposal)
        updates, new_state = line_search.update(grads, state, proposal, value_fn=value_fn, value=value)
        return (proposal + updates, new_state), (value, error)

      (proposed, _), (losses, errors) = jax.lax.scan(
        _step, init=(initial, opt_state), length=iterations
      )
      return losses, proposed, errors[-1]

    self._encode_ranges = encode_ranges
    self._decode_ranges = decode_ranges
    self.jacobian = jacobian
    self.information = information
    self.loss = loss
    self.line_search = line_search
    self._propose = jax.jit(_propose, device=self.model.device)
    self.timestamps = jnp.array(timestamps)

  def encode_conditions(self, conditions):
    return self._encode_ranges(conditions, self.condition_ranges)

  def decode_conditions(self, encoded):
    return self._decode_ranges(encoded, self.condition_ranges)

  def encode_parameters(self, parameters):
    return self._encode_ranges(parameters, self.parameter_ranges)

  def decode_parameters(self, encoded):
    return self._decode_ranges(encoded, self.parameter_ranges)

  def propose(self, key, n, controls, timestamps, parameters):
    """
    Propose n new experimental conditions.

    key:        JAX PRNG key
    n:          number of new experiments to propose
    controls:   label -> Conditions
    timestamps: label -> (n_t, )
    parameters: (n_params,)  parameter estimate in natural space

    Returns: (iterations,) loss trace, (n, n_cond) proposed encoded conditions
    """
    if len(controls) > 0:
      controls = jnp.stack([self.encode_conditions(cs) for cs in controls], axis=0)
      timestamps = jnp.stack([jnp.array(ts) for ts in timestamps], axis=0)
    else:
      controls = jnp.zeros(shape=(0, len(Conditions._fields)))
      timestamps = jnp.zeros(shape=(0, *self.timestamps.shape))

    encoded_parameters = self.encode_parameters(parameters)
    I0, error = self.information(controls, timestamps, encoded_parameters)

    assert jnp.isfinite(error) and error < self.model.tol, \
      f'high integration error {error} (tolerance = {self.model.tol})'

    m1, m2 = I0.shape
    assert m1 == m2
    I0 = I0 + self.EIGENVALUE_THRESHOLD * jnp.eye(m1)

    _, n_c = controls.shape
    key, key_initial = jax.random.split(key)
    initial = 1.0e-3 * jax.random.normal(key_initial, shape=(n, n_c))
    losses, proposed, error = self._propose(initial=initial, encoded_parameters=encoded_parameters, I0=I0)

    assert jnp.isfinite(error) and error < self.model.tol, \
      f'high integration error {error} (tolerance = {self.model.tol})'

    decoded = self.decode_conditions(proposed)

    return losses, decoded
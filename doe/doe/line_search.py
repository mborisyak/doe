from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = [
  'ArmijoLineSearch',
  'ArmijoState',
]


class ArmijoState(NamedTuple):
  pass


class ArmijoLineSearch:
  """
  Backtracking line search with Armijo (sufficient decrease) condition,
  following the optax GradientTransformation interface (init / update).

  Optionally projects iterates into a box [x_min, x_max].
  """

  def __init__(
      self,
      alpha0: float = 1.0,
      c: float = 1e-4,
      rho: float = 0.5,
      max_iters: int = 20,
      x_min: float | None = None,
      x_max: float | None = None,
  ):
    self.alpha0 = alpha0
    self.c = c
    self.rho = rho
    self.max_iters = max_iters
    self.x_min = x_min
    self.x_max = x_max

  def _project(self, x):
    if self.x_min is not None or self.x_max is not None:
      return jnp.clip(x, self.x_min, self.x_max)
    return x

  def init(self, params) -> ArmijoState:
    return ArmijoState()

  def update(self, grads, state: ArmijoState, params, *, value_fn, value=None):
    """
    grads:    gradient of loss w.r.t. params
    state:    ArmijoState
    params:   current parameter values
    value_fn: params -> scalar, evaluated for Armijo condition checks
    value:    pre-computed value_fn(params); recomputed if not provided

    Returns (updates, new_state) where new_params = params + updates.
    """
    if value is None:
      value, _ = value_fn(params)

    alpha0, c, rho, max_iters = self.alpha0, self.c, self.rho, self.max_iters
    project = self._project
    slope = jnp.sum(jnp.square(grads))

    def cond_fn(s):
      a, candidate_loss, i = s
      return ~(candidate_loss <= value - c * a * slope) & (i < max_iters)

    def body_fn(s):
      a, _, i = s
      new_a = a * rho
      value, _ = value_fn(project(params - new_a * grads))
      return new_a, value, i + 1

    init_candidate = project(params - alpha0 * grads)
    a, _, _ = jax.lax.while_loop(
      cond_fn, body_fn, (alpha0, value_fn(init_candidate)[0], 0)
    )
    new_params = project(params - a * grads)
    return new_params - params, ArmijoState()
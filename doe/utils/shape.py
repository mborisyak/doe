import math

import jax
import jax.numpy as jnp

__all__ = [
  'mesh'
]

def mesh(xs, ys):
  *batch_xs, n_x = xs.shape
  *batch_ys, n_y = ys.shape

  B_x, B_y = math.prod(batch_xs), math.prod(batch_ys)
  xs_flat = jnp.reshape(xs, shape=(B_x, n_x))
  ys_flat = jnp.reshape(ys, shape=(B_y, n_y))

  xs_flat_br = jnp.broadcast_to(xs_flat[:, None, :], shape=(B_x, B_y, n_x))
  ys_flat_br = jnp.broadcast_to(ys_flat[None, :, :], shape=(B_x, B_y, n_y))

  stacked = jnp.concatenate([xs_flat_br, ys_flat_br], axis=2)
  return jnp.reshape(stacked, shape=(*batch_xs, *batch_ys, n_x + n_y))

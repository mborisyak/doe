"""
A kernel is a scalar function `k(x, x') -> scalar`, where x, x' are (D,) points.
Lift to a Gram matrix or its diagonal with `gram` and `kdiag`. Compose with
`+`, `*` style helpers — they're just pointwise on scalars.
"""
import jax
import jax.numpy as jnp


def rbf(length_scale=1.0, variance=1.0):
  def k(x1, x2):
    sqdist = jnp.sum((x1 - x2) ** 2)
    return variance * jnp.exp(-0.5 * sqdist / length_scale ** 2)
  return k


def matern52(length_scale=1.0, variance=1.0):
  def k(x1, x2):
    s = jnp.sqrt(5.0) * jnp.linalg.norm(x1 - x2) / length_scale
    return variance * (1.0 + s + s ** 2 / 3.0) * jnp.exp(-s)
  return k


def matern32(length_scale=1.0, variance=1.0):
  def k(x1, x2):
    s = jnp.sqrt(3.0) * jnp.linalg.norm(x1 - x2) / length_scale
    return variance * (1.0 + s) * jnp.exp(-s)
  return k


def linear(variance=1.0):
  def k(x1, x2):
    return variance * jnp.dot(x1, x2)
  return k


def polynomial(degree=2, variance=1.0, bias=1.0):
  def k(x1, x2):
    return variance * (jnp.dot(x1, x2) + bias) ** degree
  return k


def sum_of(*kernels):
  def k(x1, x2):
    return sum(ki(x1, x2) for ki in kernels)
  return k


def product_of(*kernels):
  def k(x1, x2):
    out = kernels[0](x1, x2)
    for ki in kernels[1:]:
      out = out * ki(x1, x2)
    return out
  return k


def gram(k, X1, X2):
  return jax.vmap(jax.vmap(k, (None, 0)), (0, None))(X1, X2)


def kdiag(k, X):
  return jax.vmap(k)(X, X)

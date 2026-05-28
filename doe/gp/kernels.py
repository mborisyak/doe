"""
A kernel is a scalar function `k(x, x') -> scalar`, where x, x' are (D,) points.
Lift to a Gram matrix or its diagonal with `gram` and `kdiag`. Compose with
`+`, `*` style helpers — they're just pointwise on scalars.
"""
import jax
import jax.numpy as jnp


def rbf(length_scale=1.0, variance=1.0):
  """RBF kernel. `length_scale` is a scalar (isotropic) or a (D,) vector giving a
  separate scale per input dimension; each dimension is divided by its own scale."""
  ls = jnp.asarray(length_scale)
  def k(x1, x2):
    sqdist = jnp.sum(((x1 - x2) / ls) ** 2)
    return variance * jnp.exp(-0.5 * sqdist)
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
  # X1: (*A, D), X2: (*B, D) → (*A, *B). Vmap over X2's leading dims (innermost),
  # then X1's, so X1's batch axes come first in the output.
  g = k
  for _ in range(X2.ndim - 1):
    g = jax.vmap(g, (None, 0))
  for _ in range(X1.ndim - 1):
    g = jax.vmap(g, (0, None))
  return g(X1, X2)


def kdiag(k, X):
  return jax.vmap(k)(X, X)

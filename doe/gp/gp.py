"""
Gaussian Process — functional, JAX-friendly.

`GP` holds the hyper-parameters (kernel, noise) and the methods that act on
data: `fit`, `predict`, `log_marginal_likelihood`, `sample`. `State` is a
plain NamedTuple holding the fitted quantities (training inputs, Cholesky
factor, alpha) — pure data, passed back into `GP` methods.
"""
from typing import NamedTuple, Callable
from functools import partial
import math
import numpy as np
import jax
import jax.numpy as jnp

from .kernels import gram, kdiag

# Gauss-Hermite quadrature constants for ∫ f(t) N(t; m, v) dt — used by `bald`.
# Computed once at import; everything at runtime is jax. numpy is used only
# because hermgauss isn't exposed by jax.numpy.
_hg_t, _hg_w = np.polynomial.hermite.hermgauss(64)
_HG_NODES = jnp.asarray(_hg_t)
_HG_WEIGHTS = jnp.asarray(_hg_w) / jnp.sqrt(jnp.pi)


class State(NamedTuple):
  X_flat: jax.Array      # (N, D) training inputs
  L: jax.Array      # (N, N) Cholesky of K + noise * I
  alpha: jax.Array  # (N,)   L.T \ (L \ y)


class GP(object):
  def __init__(self, kernel: Callable, noise: float = 1.0e-6):
    self.kernel = kernel
    self.noise = noise

  def fit(self, X, y) -> State:
    *batch, n_x = X.shape
    N = math.prod(batch)

    assert all(bx == by for bx, by in zip(batch, y.shape))

    X_flat = jnp.reshape(X, shape=(N, n_x))
    y_flat = jnp.reshape(y, shape=(N, ))

    K = gram(self.kernel, X_flat, X_flat) + self.noise * jnp.eye(N)
    L = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((L, True), y_flat)
    return State(X_flat=X_flat, L=L, alpha=alpha)

  def predict(self, state: State, X_test, full_cov=False):
    *batch, n_x = X_test.shape
    N = math.prod(batch)
    X_test_flat = jnp.reshape(X_test, shape=(N, n_x))

    K_sx = gram(self.kernel, X_test_flat, state.X_flat)
    mean_flat = K_sx @ state.alpha
    mean = jnp.reshape(mean_flat, shape=(*batch, ))

    v = jax.scipy.linalg.solve_triangular(state.L, K_sx.T, lower=True)

    if full_cov:
      cov_flat = gram(self.kernel, X_test_flat, X_test_flat) - v.T @ v
      cov = jnp.reshape(cov_flat, shape=(*batch, *batch))

      return mean, cov
    else:
      var_flat = kdiag(self.kernel, X_test_flat) - jnp.sum(v ** 2, axis=0)
      var = jnp.reshape(var_flat, shape=(*batch,))
      return mean, var

  def batch_predict(self, state: State, X_test):
    *batch, n_test, n_x = X_test.shape
    B = math.prod(batch)
    N = state.X_flat.shape[0]
    X_test_flat = jnp.reshape(X_test, shape=(B, n_test, n_x))

    batched_gram = jax.vmap(jax.vmap(jax.vmap(self.kernel, in_axes=(0, None)), in_axes=(None, 0)), in_axes=(0, None))
    K_sx = batched_gram(X_test_flat, state.X_flat)
    mean_flat = jnp.squeeze(state.alpha[None] @ K_sx, axis=1)  # (B, n_test)

    ### (B, N, n_test)
    v_flat = jax.vmap(
      lambda L, K: jax.scipy.linalg.solve_triangular(L, K, lower=True), in_axes=(None, 0)
    )(state.L, K_sx)

    K_ss = jax.vmap(lambda X_t: gram(self.kernel, X_t, X_t))(X_test_flat)  # (B, n_test, n_test)
    # v.T @ v per batch via dot_general: contract axis 1 (N) of both, batch axis 0 (B)
    vTv = jax.lax.dot_general(v_flat, v_flat, dimension_numbers=(((1,), (1,)), ((0,), (0,))))
    cov_flat = K_ss - vTv

    mean = jnp.reshape(mean_flat, shape=(*batch, n_test, ))
    cov = jnp.reshape(cov_flat, shape=(*batch, n_test, n_test))

    return mean, cov

  def log_marginal_likelihood(self, X, y):
    state = self.fit(X, y)
    N = state.X_flat.shape[0]
    y_flat = jnp.reshape(y, shape=(N,))

    return (
      -0.5 * jnp.dot(y_flat, state.alpha)
      - jnp.sum(jnp.log(jnp.diag(state.L)))
      - 0.5 * N * jnp.log(2.0 * jnp.pi)
    )

  def sample(self, state: State, X_test, key, num_samples=1):
    *batch, n_test, n_x = X_test.shape
    B = math.prod(batch)
    mean, cov = self.batch_predict(state, X_test)
    mean_flat = jnp.reshape(mean, shape=(B, n_test))
    cov_flat = jnp.reshape(cov, shape=(B, n_test, n_test))
    L = jnp.linalg.cholesky(cov_flat)  # (B, n_test, n_test)
    z = jax.random.normal(key, shape=(num_samples, B, n_test))
    samples_flat = mean_flat[None] + jnp.einsum("bij,sbj->sbi", L, z)
    return jnp.reshape(samples_flat, shape=(num_samples, *batch, n_test))

  @staticmethod
  def fit_kernel(
      init, bounds, meta_kernel,
      X_train, y_train, X_val, y_val, noise: float, device=None
  ):
    import scipy.optimize as spopt
    _device, *_ = jax.devices(device)

    @partial(jax.jit, device=_device)
    def loss(params) -> jax.Array:
      kernel = meta_kernel(params)
      gp = GP(kernel, noise=noise)
      negative_log_likelihood = gp.log_marginal_likelihood(X_train, y_train)
      return jnp.mean(negative_log_likelihood)

    loss_and_grad = jax.jit(jax.value_and_grad(loss, argnums=0), device=_device)

    result = spopt.minimize(loss_and_grad, init, method='L-BFGS-B', bounds=bounds, jac=True)

    if not result.success:
      raise ValueError(f'Optimization failed: {result.message}')

    return result.x
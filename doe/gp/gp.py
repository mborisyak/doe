"""
Gaussian Process — functional, JAX-friendly.

`GP` holds the hyper-parameters (kernel, noise) and the methods that act on
data: `fit`, `predict`, `log_marginal_likelihood`, `sample`. `State` is a
plain NamedTuple holding the fitted quantities (training inputs, Cholesky
factor, alpha) — pure data, passed back into `GP` methods.
"""
from typing import NamedTuple, Callable
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
  X: jax.Array      # (N, D) training inputs
  L: jax.Array      # (N, N) Cholesky of K + noise * I
  alpha: jax.Array  # (N,)   L.T \ (L \ y)


class GP(object):
  def __init__(self, kernel: Callable, noise: float = 1.0e-6):
    self.kernel = kernel
    self.noise = noise

  def fit(self, X, y) -> State:
    N = X.shape[0]
    K = gram(self.kernel, X, X) + self.noise * jnp.eye(N)
    L = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((L, True), y)
    return State(X=X, L=L, alpha=alpha)

  def predict(self, state: State, X_test, full_cov=False):
    K_sx = gram(self.kernel, X_test, state.X)
    mean = K_sx @ state.alpha
    v = jax.scipy.linalg.solve_triangular(state.L, K_sx.T, lower=True)

    if full_cov:
      cov = gram(self.kernel, X_test, X_test) - v.T @ v
      return mean, cov
    else:
      var = kdiag(self.kernel, X_test) - jnp.sum(v ** 2, axis=0)
      return mean, var

  def log_marginal_likelihood(self, state: State, y):
    N = state.X.shape[0]
    return (
      -0.5 * jnp.dot(y, state.alpha)
      - jnp.sum(jnp.log(jnp.diag(state.L)))
      - 0.5 * N * jnp.log(2.0 * jnp.pi)
    )

  @staticmethod
  def prob_below(mean, std, threshold):
    return jax.scipy.stats.norm.cdf(threshold, loc=mean, scale=std)

  @staticmethod
  def prob_above(mean, std, threshold):
    return jax.scipy.stats.norm.sf(threshold, loc=mean, scale=std)

  @staticmethod
  def cross_entropy(mean1, var1, mean2, var2, threshold=0.0):
    """Binary cross-entropy between the Bernoulli classifiers induced by
    P(y > threshold) under N(mean1, var1) and N(mean2, var2)."""
    p = jax.scipy.stats.norm.sf(threshold, loc=mean1, scale=jnp.sqrt(var1))
    q = jax.scipy.stats.norm.sf(threshold, loc=mean2, scale=jnp.sqrt(var2))
    return -p * jnp.log(q) - (1.0 - p) * jnp.log(1.0 - q)

  def bald(self, mean, var, target, eps):
    """BALD acquisition for the tube indicator I(x) = 1{|f(x) - target| < eps}.

    Returns MI(I(x); y(x)) under the GP posterior with observation noise
    self.noise — i.e., expected reduction in binary entropy of "is x inside
    the eps-tube around target" if we observed y at x.

    Houlsby et al. 2011, "Bayesian Active Learning for Classification and
    Preference Learning," arXiv:1112.5745. The level-set / tube application
    is also closely related to Bect et al. 2012, Stat. Comput. 22(3).
    """
    v, s = var, self.noise
    sigma = jnp.sqrt(v)
    delta = target - mean

    p_now = (jax.scipy.stats.norm.cdf((delta + eps) / sigma)
             - jax.scipy.stats.norm.cdf((delta - eps) / sigma))
    H_now = jax.scipy.special.entr(p_now) + jax.scipy.special.entr(1.0 - p_now)

    # After hypothetical noisy observation y(x):
    #   posterior var  σ²_new = v*s/(v+s)            (deterministic shrink)
    #   posterior mean μ_new ~ N(mean, v² / (v+s))   (Gaussian over hypothetical y)
    var_mu = v ** 2 / (v + s)
    sigma_new = jnp.sqrt(v * s / (v + s))

    mean_a = jnp.asarray(mean)[..., None]
    target_a = jnp.asarray(target)[..., None]
    sigma_new_a = jnp.asarray(sigma_new)[..., None]
    scale_a = jnp.sqrt(2.0 * jnp.asarray(var_mu))[..., None]

    mu_new = mean_a + scale_a * _HG_NODES
    delta_new = target_a - mu_new
    p_new = (jax.scipy.stats.norm.cdf((delta_new + eps) / sigma_new_a)
             - jax.scipy.stats.norm.cdf((delta_new - eps) / sigma_new_a))
    H_new = jax.scipy.special.entr(p_new) + jax.scipy.special.entr(1.0 - p_new)
    H_expected = jnp.sum(_HG_WEIGHTS * H_new, axis=-1)

    return H_now - H_expected

  def sample(self, state: State, X_test, key, num_samples=1, jitter=1.0e-6):
    mean, cov = self.predict(state, X_test, full_cov=True)
    cov = cov + jitter * jnp.eye(X_test.shape[0])
    L = jnp.linalg.cholesky(cov)
    z = jax.random.normal(key, shape=(num_samples, X_test.shape[0]))
    return mean[None, :] + z @ L.T
"""
Discriminative Design of Experiments for a Gaussian-process model.

Goal: pick measurement batches that best resolve the sign pattern
``[f(x_i) > threshold]`` over a grid ``G`` of points, for a GP model of a scalar
field ``f``. The exact expected information gain about the ``2**|G|`` sign
pattern is intractable, so we score the *expected posterior* uncertainty over
``G`` with cheap proxies and optimise the whole batch continuously by gradient
descent.

Key fact exploited throughout: the GP posterior covariance on ``G`` after adding
a batch ``B`` is value-independent -- only the posterior mean moves with the
(unknown) batch outcome. So the outcome enters only through reparameterised mean
draws ``m0 + (z @ chol(M)^T) @ A`` with fixed standard normals ``z`` (common
random numbers), which keeps the acquisition smooth and differentiable in ``B``.

Proxies (``PROXIES`` maps name -> kind; ``m`` is the threshold-centred posterior
mean ``E[f] - threshold`` on the grid, ``cov`` the posterior grid covariance):

  ``marginal``        ``Sum_i H_b(Phi(m_i / sigma_i))`` -- marginal sign entropy
  ``marginal_eigen``  marginal sign entropy in the covariance eigenbasis
                      (decorrelated coordinates; transformed mean ``Q^T m``)
  ``soft_rank_Mg``    effective (soft) rank of ``cov + m m^T``
  ``modulated``       soft rank of weights ``var_k / (var_k + (Q^T m)_k^2)``
  ``effective_rank``  effective rank of ``cov`` (mean-blind)

Entropy proxies (``marginal*``) average the heuristic over the batch outcome;
rank proxies (covariance-driven) are evaluated at the expected posterior mean.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
import scipy.optimize as spopt

from ..gp.kernels import gram

__all__ = [
  'DiscriminativeDoE',
  'optimize_batch',
  'PROXIES',
]

# name -> kind. The kind tells the engine how to take the outcome-expectation.
PROXIES = {
  'marginal': 'marginal',
  'marginal_eigen': 'marginal_eigen',
  'soft_rank_Mg': 'rank_Mg',
  'modulated': 'rank_mod',
  'effective_rank': 'rank_eff',
}


def _bits(p):
  # binary entropy in bits; eps must be float32-representable (1 - 1e-9 == 1f).
  p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
  return -(p * jnp.log(p) + (1.0 - p) * jnp.log1p(-p)) / jnp.log(2.0)


def _effrank(eig):
  # exp(entropy of the normalised eigenvalues): a smooth, differentiable rank.
  eig = jnp.clip(eig, 1e-12, None)
  p = eig / jnp.sum(eig)
  return jnp.exp(-jnp.sum(p * jnp.log(p)))


def _softrank(w):
  w = jnp.clip(w, 1e-12, None)
  p = w / jnp.sum(w)
  return jnp.exp(-jnp.sum(p * jnp.log(p)))


def _expected_score(kind, m0, cov_post, A, Lm, z):
  """Expected proxy over the unknown batch outcome. ``cov_post`` is the
  (value-independent) posterior grid covariance after conditioning on the batch;
  ``m0`` the threshold-centred prior-to-this-step grid mean; mean draws are
  ``m0 + (z @ Lm^T) @ A`` with fixed standard normals ``z``."""
  cov_post = 0.5 * (cov_post + cov_post.T)
  if kind == 'rank_eff':
    return _effrank(jnp.linalg.eigvalsh(cov_post))
  if kind == 'rank_Mg':
    return _effrank(jnp.linalg.eigvalsh(cov_post + jnp.outer(m0, m0)))
  if kind == 'rank_mod':
    lam, Q = jnp.linalg.eigh(cov_post)
    var = jnp.clip(lam, 0.0, None)
    mu_rot = Q.T @ m0
    return _softrank(var / (var + mu_rot ** 2 + 1e-12))
  mean_draws = m0[None, :] + (z @ Lm.T) @ A                # (n_outer, n_g), centred
  if kind == 'marginal_eigen':
    lam, Q = jnp.linalg.eigh(cov_post)
    sigma = jnp.sqrt(jnp.clip(lam, 1e-12, None))
    mu_rot = mean_draws @ Q
    return jnp.mean(jnp.sum(_bits(ndtr(mu_rot / sigma[None, :])), axis=1))
  s = jnp.sqrt(jnp.clip(jnp.diag(cov_post), 1e-12, None))
  return jnp.mean(jnp.sum(_bits(ndtr(mean_draws / s[None, :])), axis=1))


def _grid_blocks(gp, state, X_grid):
  """Grid posterior mean and covariance under the current GP (independent of any
  future batch). Handles the zero-data (prior) state."""
  k = gp.kernel
  K_gg = gram(k, X_grid, X_grid)
  if state.X_flat.shape[0] > 0:
    K_tg = gram(k, state.X_flat, X_grid)
    vg = jax.scipy.linalg.solve_triangular(state.L, K_tg, lower=True)
    mean_g = K_tg.T @ state.alpha
    S_gg = K_gg - vg.T @ vg
    return mean_g, S_gg, vg
  return jnp.zeros(X_grid.shape[0]), K_gg, None


def optimize_batch(gp, state, X_grid, threshold, batch_size, bounds, key,
                   proxy='marginal', n_outer=24, n_restarts=4, noise=None):
  """Continuously optimise a whole batch ``B`` (``batch_size`` x dim) over the box
  ``bounds`` (list of ``(lo, hi)`` per input dim) to MAXIMISE the expected
  information gain of ``proxy`` about the grid sign pattern, via multi-start
  L-BFGS-B with JAX gradients.

  Returns ``(B, eig)``: the chosen batch and ``score_prior - min_score``
  (bits for entropy proxies, soft-rank units for rank proxies)."""
  kind = PROXIES[proxy]
  noise = gp.noise if noise is None else noise
  dim = X_grid.shape[1]
  k = gp.kernel
  has_data = state.X_flat.shape[0] > 0

  mean_g, S_gg, vg = _grid_blocks(gp, state, X_grid)
  m0 = mean_g - threshold
  z = jax.random.normal(key, (n_outer, batch_size))
  score_prior = float(_expected_score(
    kind, m0, S_gg, jnp.zeros((batch_size, X_grid.shape[0])),
    jnp.eye(batch_size), jnp.zeros((n_outer, batch_size))))

  @jax.jit
  def neg_eig(b_flat):
    B = b_flat.reshape(batch_size, dim)
    K_gB = gram(k, X_grid, B)
    K_BB = gram(k, B, B)
    if has_data:
      K_tB = gram(k, state.X_flat, B)
      vB = jax.scipy.linalg.solve_triangular(state.L, K_tB, lower=True)
      S_gB = K_gB - vg.T @ vB
      S_BB = K_BB - vB.T @ vB
    else:
      S_gB, S_BB = K_gB, K_BB
    M = S_BB + noise * jnp.eye(batch_size)
    Lm = jnp.linalg.cholesky(M)
    A = jax.scipy.linalg.cho_solve((Lm, True), S_gB.T)
    cov_post = S_gg - S_gB @ A
    return _expected_score(kind, m0, cov_post, A, Lm, z)

  vng = jax.jit(jax.value_and_grad(neg_eig))
  def fun(x):
    v, g = vng(jnp.asarray(x, jnp.float32))
    return float(v), np.asarray(g, np.float64)

  lo = jnp.asarray([b[0] for b in bounds])
  hi = jnp.asarray([b[1] for b in bounds])
  sp_bounds = [bounds[d] for _ in range(batch_size) for d in range(dim)]
  best_x, best_f = None, np.inf
  for kk in jax.random.split(key, n_restarts):
    b0 = jax.random.uniform(kk, (batch_size, dim), minval=lo, maxval=hi).reshape(-1)
    res = spopt.minimize(fun, np.asarray(b0, np.float64), jac=True,
                         method='L-BFGS-B', bounds=sp_bounds)
    if res.fun < best_f:
      best_f, best_x = res.fun, res.x
  return jnp.asarray(best_x.reshape(batch_size, dim), jnp.float32), score_prior - best_f


class DiscriminativeDoE:
  """Discriminative DoE for a GP model of ``f``, targeting the sign pattern
  ``[f(x) > threshold]`` over ``X_grid``.

  >>> doe = DiscriminativeDoE(gp, X_grid, threshold=0.5, proxy='marginal')
  >>> B, eig = doe.suggest(state, batch_size=8, bounds=[(0, 1), (0, 1)], key=key)
  >>> q = doe.sign_probability(state)          # P(f > threshold) on the grid
  """

  def __init__(self, gp, X_grid, threshold, proxy='marginal', *,
               n_outer=24, n_restarts=4):
    if proxy not in PROXIES:
      raise ValueError(f'unknown proxy {proxy!r}; choose from {list(PROXIES)}')
    self.gp = gp
    self.X_grid = jnp.asarray(X_grid)
    self.threshold = float(threshold)
    self.proxy = proxy
    self.n_outer = n_outer
    self.n_restarts = n_restarts

  def suggest(self, state, batch_size, bounds, key):
    """Optimise and return ``(B, eig)``: the next batch and its expected info gain."""
    return optimize_batch(
      self.gp, state, self.X_grid, self.threshold, batch_size, bounds, key,
      proxy=self.proxy, n_outer=self.n_outer, n_restarts=self.n_restarts)

  def sign_probability(self, state):
    """Posterior ``P(f > threshold)`` at each grid point (latent, noise-free)."""
    mean, var = self.gp.predict(state, self.X_grid)
    sigma = jnp.sqrt(jnp.clip(var, 1e-12, None))
    return ndtr((mean - self.threshold) / sigma)

  def accuracy(self, state, f_true):
    """Point-wise sign accuracy that accounts for GP uncertainty:
    ``mean_i [ t_i q_i + (1 - t_i)(1 - q_i) ]`` with ``q_i = P(f_i > threshold)``
    and true labels ``t_i = 1[f_true_i > threshold]``."""
    q = np.asarray(self.sign_probability(state), float)
    t = (np.asarray(f_true, float) > self.threshold).astype(float)
    return float(np.mean(t * q + (1.0 - t) * (1.0 - q)))

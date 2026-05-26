"""
Discriminative Design of Experiments for a Gaussian-process model.

Goal: pick measurement batches that best resolve the sign pattern
``[f(x_i) > threshold]`` over a grid ``G`` of points, for a GP model of a scalar
field ``f``. The exact expected information gain about the ``2**|G|`` sign
pattern is intractable, so we score the *expected posterior* uncertainty over
``G`` with the **marginal sign-entropy** proxy and optimise the whole batch
continuously by gradient descent::

    H_marg(m, sigma) = Sum_i H_b(Phi(m_i / sigma_i)),

``m`` the threshold-centred posterior grid mean, ``sigma_i`` the per-point
posterior standard deviation. Across the stress tests this was the proxy whose
information-gain ordering tracked the exact entropy on both sharp and subtle
boundaries; richer covariance-spectrum proxies (eigen / soft-rank / GP
differential-entropy variants) did not robustly beat it and were dropped.

Key fact exploited throughout: the GP posterior covariance on ``G`` after adding
a batch ``B`` is value-independent -- only the posterior mean moves with the
(unknown) batch outcome. The outcome therefore enters only through reparameterised
mean draws ``m0 + (z @ chol(M)^T) @ A`` with fixed standard normals ``z`` (common
random numbers), which keeps the acquisition smooth and differentiable in ``B``.

Because the proxy depends on the grid covariance only through its **diagonal**,
the engine never forms the ``n_g x n_g`` grid covariance and never takes an
eigendecomposition: per-point variances are built directly from ``kdiag`` minus
the low-rank reductions, so cost and memory scale as ``O(n_g x batch_size)``.

Batch optimisation runs entirely on-device: ``n_multi_start`` random restarts are
optimised *in parallel* (one vmapped batch, distinct from the ``batch_size``
experiment batch) by optax AdamW inside a jitted ``lax.scan`` of fixed length,
projected back into the box each step; the best start is returned.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
import optax

from ..gp.kernels import gram, kdiag

__all__ = [
  'DiscriminativeDoE',
  'optimize_batch',
  'PROXIES',
]

# AdamW schedule for the on-device batch optimisation.
N_STEPS = 200
LR = 0.05
N_MULTI_START = 8

# Only the marginal sign-entropy proxy remains; kept as a registry so the proxy
# name is validated and the CLI/API stay stable.
PROXIES = {'marginal'}


def _bits(p):
  # binary entropy in bits; eps must be float32-representable (1 - 1e-9 == 1f).
  p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
  return -(p * jnp.log(p) + (1.0 - p) * jnp.log1p(-p)) / jnp.log(2.0)


def _grid_prior(gp, state, X_grid):
  """Grid posterior MEAN and per-point VARIANCE (the diagonal of the grid
  covariance) under the current GP -- the full ``n_g x n_g`` covariance is never
  formed. Also returns ``vg = L^-1 K(train, grid)`` (``N x n_g``, used for the
  batch conditioning), or ``None`` at the zero-data prior."""
  k = gp.kernel
  var_g = kdiag(k, X_grid)                                  # (n_g,) prior diagonal
  if state.X_flat.shape[0] > 0:
    K_tg = gram(k, state.X_flat, X_grid)                    # (N, n_g)
    vg = jax.scipy.linalg.solve_triangular(state.L, K_tg, lower=True)   # (N, n_g)
    mean_g = K_tg.T @ state.alpha                           # (n_g,)
    var_g = var_g - jnp.sum(vg ** 2, axis=0)                # (n_g,) posterior diagonal
    return mean_g, var_g, vg
  return jnp.zeros(X_grid.shape[0]), var_g, None


def optimize_batch(gp, state, X_grid, threshold, batch_size, bounds, key,
                   proxy='marginal', *, n_outer=24, n_multi_start=N_MULTI_START,
                   n_steps=N_STEPS, lr=LR, noise=None):
  """Continuously optimise a whole batch ``B`` (``batch_size`` x dim) over the box
  ``bounds`` (list of ``(lo, hi)`` per input dim) to MAXIMISE the expected
  marginal-sign-entropy information gain about the grid sign pattern.

  All ``n_multi_start`` random restarts are optimised in parallel by optax AdamW
  inside a jitted ``lax.scan`` of ``n_steps`` steps, each step projected back into
  the box. Only per-point grid variances are used, so no ``n_g x n_g`` covariance
  or eigendecomposition is ever computed. Runs on whatever device JAX is
  configured for (GPU/CPU).

  Returns ``(B, eig)``: the best batch and ``score_prior - min_score`` (bits)."""
  if proxy not in PROXIES:
    raise ValueError(f'unknown proxy {proxy!r}; choose from {list(PROXIES)}')
  noise = gp.noise if noise is None else noise
  dim = X_grid.shape[1]
  k = gp.kernel
  has_data = state.X_flat.shape[0] > 0

  mean_g, var_g, vg = _grid_prior(gp, state, X_grid)
  m0 = mean_g - threshold
  s_prior = jnp.sqrt(jnp.clip(var_g, 1e-12, None))
  z = jax.random.normal(key, (n_outer, batch_size))
  # zero-batch score: posterior == prior, mean draws collapse to m0.
  score_prior = float(jnp.sum(_bits(ndtr(m0 / s_prior))))

  lo = jnp.asarray([b[0] for b in bounds], jnp.float32)
  hi = jnp.asarray([b[1] for b in bounds], jnp.float32)

  def neg_eig(B):                                 # B: (batch_size, dim) -> scalar
    K_gB = gram(k, X_grid, B)                     # (n_g, nb)
    K_BB = gram(k, B, B)                          # (nb, nb)
    if has_data:
      K_tB = gram(k, state.X_flat, B)             # (N, nb)
      vB = jax.scipy.linalg.solve_triangular(state.L, K_tB, lower=True)   # (N, nb)
      S_gB = K_gB - vg.T @ vB                     # (n_g, nb)
      S_BB = K_BB - vB.T @ vB                     # (nb, nb)
    else:
      S_gB, S_BB = K_gB, K_BB
    M = S_BB + noise * jnp.eye(batch_size)
    Lm = jnp.linalg.cholesky(M)
    A = jax.scipy.linalg.cho_solve((Lm, True), S_gB.T)            # (nb, n_g)
    reduction = jnp.clip(jnp.sum(S_gB * A.T, axis=1), 0.0, None)  # (n_g,) diag only
    s_post = jnp.sqrt(jnp.clip(var_g - reduction, 1e-12, None))   # (n_g,)
    mean_draws = m0[None, :] + (z @ Lm.T) @ A                     # (n_outer, n_g)
    return jnp.mean(jnp.sum(_bits(ndtr(mean_draws / s_post[None, :])), axis=1))

  vgrad = jax.value_and_grad(neg_eig)
  # weight_decay=0: AdamW's decoupled decay would pull the design toward 0 (a box
  # corner) and bias the result; with it off the step is scale-equivariant.
  opt = optax.adamw(lr, weight_decay=0.0)
  B0 = jax.random.uniform(key, (n_multi_start, batch_size, dim),
                          minval=lo, maxval=hi)

  @jax.jit
  def run(B0):
    opt_state = opt.init(B0)

    def step(carry, _):
      B, st = carry
      _, grads = jax.vmap(vgrad)(B)               # per-start value & gradient
      updates, st = opt.update(grads, st, B)
      B = jnp.clip(optax.apply_updates(B, updates), lo, hi)   # project to box
      return (B, st), None

    (B_final, _), _ = jax.lax.scan(step, (B0, opt_state), None, length=n_steps)
    return B_final, jax.vmap(neg_eig)(B_final)

  B_final, final_vals = run(B0)
  best = int(jnp.argmin(final_vals))
  return B_final[best], score_prior - float(final_vals[best])


class DiscriminativeDoE:
  """Discriminative DoE for a GP model of ``f``, targeting the sign pattern
  ``[f(x) > threshold]`` over ``X_grid`` via the marginal sign-entropy proxy.

  >>> doe = DiscriminativeDoE(gp, X_grid, threshold=0.5)
  >>> B, eig = doe.suggest(state, batch_size=8, bounds=[(0, 1), (0, 1)], key=key)
  >>> q = doe.sign_probability(state)          # P(f > threshold) on the grid
  """

  def __init__(self, gp, X_grid, threshold, proxy='marginal', *,
               n_outer=24, n_multi_start=N_MULTI_START, n_steps=N_STEPS, lr=LR):
    if proxy not in PROXIES:
      raise ValueError(f'unknown proxy {proxy!r}; choose from {list(PROXIES)}')
    self.gp = gp
    self.X_grid = jnp.asarray(X_grid)
    self.threshold = float(threshold)
    self.proxy = proxy
    self.n_outer = n_outer
    self.n_multi_start = n_multi_start
    self.n_steps = n_steps
    self.lr = lr

  def suggest(self, state, batch_size, bounds, key):
    """Optimise and return ``(B, eig)``: the next batch and its expected info gain."""
    return optimize_batch(
      self.gp, state, self.X_grid, self.threshold, batch_size, bounds, key,
      proxy=self.proxy, n_outer=self.n_outer, n_multi_start=self.n_multi_start,
      n_steps=self.n_steps, lr=self.lr)

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
